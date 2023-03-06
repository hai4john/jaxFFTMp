
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cufftMp.h>
#include <cuda.h>
#include <mpi.h>

namespace jaxfftmp
{
    #define CUDA_CHECK(ans) { gpu_checkAssert((ans), __FILE__, __LINE__); }
    inline void gpu_checkAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr,"CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    #define CUFFT_CHECK(ans) { cufft_check((ans), __FILE__, __LINE__); }
    inline void cufft_check(int code, const char *file, int line, bool abort=true)
    {
        if (code != CUFFT_SUCCESS)
        {
            fprintf(stderr,"CUFFT_CHECK: %d %s %d\n", code, file, line);
            if (abort) exit(code);
        }
    }
    // https://en.cppreference.com/w/cpp/numeric/bit_cast
    template <class To, class From>
    typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                              std::is_trivially_copyable<To>::value,
                          To>::type
    bit_cast(const From &src) noexcept
    {
        static_assert(
            std::is_trivially_constructible<To>::value,
            "This implementation additionally requires destination type to be trivially constructible");

        To dst;
        memcpy(&dst, &src, sizeof(To));
        return dst;
    }

    template <typename T>
     pybind11::capsule EncapsulateFunction(T *fn)
    {
        return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
    }

    // alloc mem to storage stream structure, initialize stream
    void* initStream()
    {
        cudaStream_t stream;

        CUDA_CHECK(cudaStreamCreate(&stream));

        printf("+++++++++1+++++++ %x\n", stream);
        return static_cast<void*>(stream);
    }

	// Initialize plans(create plans, attach the MPI communicator to plans, Set the stream, make plans）
    int initPlan(void* stream, int nx, int ny, int nz, cufftType fftType)
    {
        // Initialize plans and stream
        cufftHandle plan = 0;
        MPI_Comm comm = MPI_COMM_WORLD;

        CUFFT_CHECK(cufftCreate(&plan));

        // Attach the MPI communicator to the plans
        CUFFT_CHECK(cufftMpAttachComm(plan, CUFFT_COMM_MPI, &comm));

        // Set the stream
        CUFFT_CHECK(cufftSetStream(plan, (cudaStream_t)(stream)));

        // Make the plan
        size_t workspace;
        CUFFT_CHECK(cufftMakePlan3d(plan, nx, ny, nz, fftType, &workspace));

        return  (int)(plan);
    }

    void freeDesc(void* desc_str)
    {
        cudaLibXtDesc *desc = (cudaLibXtDesc*)desc_str;

        CUFFT_CHECK(cufftXtFree(desc));
    }

    void freePlan(int plan)
    {
        CUFFT_CHECK(cufftDestroy((cufftHandle)plan));
    }

    void freeStream(void* stream)
    {
        printf("+++++++++2+++++++ %x\n", stream);
        CUDA_CHECK(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
    }

	// Run cuFFT (allocate GPU memory, copy CPU data to GPU, exec descriptor) 
	// mutil version ( c2c/ r2c/ c2r and build-in stlabs/ custom slabs[pencils or boxes])
	// rfft3d, irfft3d
    // attention: rfft3d, irfft3d use different plan (plan handle)
    void* rfft3d(int plan, int nx, int ny, int nz, float* cpu_data, const int rank, const int size)
    {
        cufftHandle plan_r2c = (cufftHandle)plan;
        cudaLibXtDesc *desc;

        CUFFT_CHECK(cufftXtMalloc(plan_r2c, &desc, CUFFT_XT_FORMAT_INPLACE));
        CUFFT_CHECK(cufftXtMemcpy(plan_r2c, (void*)desc, (void*)cpu_data, CUFFT_COPY_HOST_TO_DEVICE));

        // Run R2C
        CUFFT_CHECK(cufftXtExecDescriptor(plan_r2c, desc, desc, CUFFT_FORWARD));

        // At this point, data is distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED
        // This applies an element-wise scaling function to the GPU data located in desc->descriptor->data[0]
        /*
        auto [begin_d, end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_R2C,
                                             rank, size, nx, ny, nz, (cufftComplex*)desc->descriptor->data[0]);
        const size_t num_elements = std::distance(begin_d, end_d);
        const size_t num_threads  = 128;
        const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
        scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz);
        */

        return  (void*)desc;
    }

    void irfft3d(int plan, void* desc_str,void* stream_p, float* cpu_data)
    {
        cufftHandle plan_c2r = (cufftHandle)plan;
        cudaLibXtDesc *desc = (cudaLibXtDesc*)desc_str;
        cudaStream_t stream = (cudaStream_t)(*(cudaStream_t*)stream_p);

        // Run C2R
        CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE));

        // Copy back to CPU and free
        // Data is again distributed according to CUFFT_XT_FORMAT_INPLACE
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        CUFFT_CHECK(cufftXtMemcpy(plan_c2r, (void*)cpu_data, (void*)desc, CUFFT_COPY_DEVICE_TO_HOST));
    }

    void rfft3d_wrapper(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len)
    {

    }

    void irfft3d_wrapper(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len)
    {

    }

    // Utility to export ops to XLA
    pybind11::dict Registrations()
    {
        pybind11::dict dict;

        dict["rfft3d_wrapper"] = EncapsulateFunction(rfft3d_wrapper);
        dict["irfft3d_wrapper"] = EncapsulateFunction(irfft3d_wrapper);
        return dict;
    }
}

namespace py = pybind11;
namespace jf = jaxfftmp;

PYBIND11_MODULE(_cufftMp, m)
{
    // Exported types
    py::enum_<cufftType_t>(m, "CufftType")
        .value("CUFFT_R2C", cufftType_t::CUFFT_R2C)
        .value("CUFFT_C2R", cufftType_t::CUFFT_C2R)
        .value("CUFFT_C2C", cufftType_t::CUFFT_C2C)
        .value("CUFFT_D2Z", cufftType_t::CUFFT_D2Z)
        .value("CUFFT_Z2D", cufftType_t::CUFFT_Z2D)
        .value("CUFFT_Z2Z", cufftType_t::CUFFT_Z2Z)
        .export_values();

    // Exported functions
    m.def("init_stream", &jf::initStream);
    m.def("init_plan", &jf::initPlan);
    m.def("free_desc", &jf::freeDesc);
    m.def("free_plan", &jf::freePlan);
    m.def("free_stream", &jf::freeStream);
    m.def("rfft3d", &jf::rfft3d);
    m.def("irfft3d", &jf::irfft3d);

    // Function registering the custom ops
    m.def("registrations", &jf::Registrations);
}





