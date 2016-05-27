/*==========================================================================

  Copyright (c) 2016 Uditha L. Jayarathne, ujayarat@robarts.ca
  << This was based on the original implementation by Nick Lamprianidis >> 

  Use, modification and redistribution of the software, in source or
  binary forms, are permitted provided that the following terms and
  conditions are met:

  1) Redistribution of the source code, in verbatim or modified
  form, must retain the above copyright notice, this license,
  the following disclaimer, and any notices that refer to this
  license and/or the following disclaimer.  

  2) Redistribution in binary form must include the above copyright
  notice, a copy of this license and the following disclaimer
  in the documentation or with other materials provided with the
  distribution.

  3) Modified copies of the source code must be clearly marked as such,
  and must not be misrepresented as verbatim copies of the source code.

  THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
  WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
  MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
  OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
  THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGES.
  =========================================================================*/

#ifndef VTKCLUTILS_HPP
#define VTKCLUTILS_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <chrono>
#include <cassert>
#include <cmath>

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

/*! \brief It brings together functionality common to all OpenCL projects.
 *  
 *  It offers structures that aim to ease the process of setting up and 
 *  maintaining an OpenCL environment.
 */
namespace vtkclutils
{
    /*! \brief Returns the name of an error code. */
    const char* GetOpenCLErrorCodeString (int errorCode);

    /*! \brief Checks the availability of the "GL Sharing" capability. */
    bool CheckCLGLInterop (cl::Device &device);

    /*! \brief Reads in the contents from the requested files. */
    void ReadSource (const std::vector<std::string> &kernel_filenames, 
                     std::vector<std::string> &sourceCodes);

    /*! \brief Splits a string on the requested delimiter. */
    void Split (const std::string &str, char delim, 
                std::vector<std::string> &names);


    /*! \brief Creates a pair of a char array (source code) and its size. */
    std::pair<const char *, size_t> 
        make_kernel_pair (const std::string &kernel_filename);


    /*! \brief Sets up an OpenCL environment.
     *  \details Prepares the essential OpenCL objects for the execution of 
     *           kernels. This class aims to allow rapid prototyping by hiding 
     *           away all the boilerplate code necessary for establishing 
     *           an OpenCL environment.
     */
    class vtkCLEnv
    {
    public:
        vtkCLEnv (const std::vector<std::string> &kernel_filenames = std::vector<std::string> (), 
               const char *build_options = nullptr); 
        vtkCLEnv (const std::string &kernel_filename, 
               const char *build_options = nullptr);
        virtual ~vtkCLEnv () {};
        /*! \brief Gets back one of the existing contexts. */
        cl::Context& GetContext (unsigned int pIdx = 0);
        /*! \brief Gets back one of the existing command queues 
         *         in the specified context. */
        cl::CommandQueue& GetQueue (unsigned int ctxIdx = 0, unsigned int qIdx = 0);
        /*! \brief Gets back one of the existing programs. */
        cl::Program& GetProgram (unsigned int pgIdx = 0);
        /*! \brief Gets back one of the existing kernels in some program. */
        cl::Kernel& GetKernel (const char *kernelName, unsigned int pgIdx = 0);
        /*! \brief Creates a context for all devices in the requested platform. */
        cl::Context& AddContext (unsigned int pIdx, const bool gl_shared = false);
        /*! \brief Creates a queue for the specified device in the specified context. */
        cl::CommandQueue& AddQueue (unsigned int ctxIdx, unsigned int dIdx, cl_command_queue_properties props = 0);
        /*! \brief Creates a queue for the GL-shared device in the specified context. */
        cl::CommandQueue& AddQueueGL (unsigned int ctxIdx, cl_command_queue_properties props = 0);
        /*! \brief Creates a program for the specified context. */
        cl::Kernel& AddProgram (unsigned int ctxIdx, 
                                const std::vector<std::string> &kernel_filenames, 
                                const char *kernel_name = nullptr, 
                                const char *build_options = nullptr);
        cl::Kernel& AddProgram (unsigned int ctxIdx, 
                                const std::string &kernel_filename, 
                                const char *kernel_name = nullptr, 
                                const char *build_options = nullptr);

        // Objects associated with an OpenCL environment.
        // For each of a number of objects, there is a vector that 
        // can hold all instances of that object.

        std::vector<cl::Platform> platforms;  /*!< List of platforms. */
        /*! \brief List of devices per platform.
         *  \details Holds a vector of devices per platform. */
        std::vector< std::vector<cl::Device> > devices;

    private:
        std::vector<cl::Context> contexts;  /*!< List of contexts. */
        /*! \brief List of queues per context.
         *  \details Holds a vector of queues per context. */
        std::vector< std::vector<cl::CommandQueue> > queues;
        std::vector<cl::Program> programs;  /*!< List of programs. */
        /*! \brief List of kernels per program.
         *  \details Holds a vector of kernels per program. */
        std::vector< std::vector<cl::Kernel> > kernels;

    protected:
        /*! \brief Initializes the OpenGL memory buffers.
         *  \details If CL-GL interop is desirable, CLEnv has to be derived and
         *           `initGLMemObjects` be implemented. `initGLMemObjects` will 
         *           have to create all necessary OpenGL memory buffers.
         *  \note Setting up CL-GL interop requires the following procedure:
         *        (i) Initialize OpenGL context, (ii) Initilize OpenCL context,
         *        (iii) Create OpenGL buffers, (iv) Create OpenCL buffers.
         *  \note Do not call `initGLMemObjects` directly. `initGLMemObjects`  
         *        will be called by `addContext` when it is asked for a 
         *        GL-shared CL context to be created.
         */
        virtual void initGLMemObjects () {};

    private:
        /*! \brief Maps kernel names to kernel indices.
         *         There is one unordered_map for every program.
         *  
         *  For every program in programs, there is an element in kernelIdx.
         *  For every kernel in program i, there is a mapping from the kernel 
         *  name to the kernel index in kernels[i].
         */
        std::vector< std::unordered_map<std::string, unsigned int> > kernelIdx;
    };


    /*! \brief Facilitates the conveyance of `CLEnv` arguments.
     *  \details `CLEnv` creates an OpenCL environment. A `CLEnv` object 
     *           potentially contains many platforms, contexts, queues, etc, 
     *           that are to be used by different (independent) subsystems. 
     *           Those subsystems will have to know where to look inside CLEnv 
     *           for their associated CL objects. `CLEnvInfo` organizes this 
     *           process of information transfer between OpenCL systems.
     *           
     *  \tparam nQueues the number of command queue indices to be held by `CLEnvInfo`.
     */
    template<unsigned int nQueues = 1>
    class vtkCLEnvInfo
    {
    public:
        /*! \brief Initializes a `CLEnvInfo` object.
         *  \details All provided indices are supposed to follow the order the 
         *           associated objects were created in the associated `CLEnv` instance.
         *           
         *  \param[in] _pIdx platform index.
         *  \param[in] _dIdx device index.
         *  \param[in] _ctxIdx context index.
         *  \param[in] _qIdx vector with command queue indices.
         *  \param[in] _pgIdx program index.
         */
        vtkCLEnvInfo (unsigned int _pIdx = 0, unsigned int _dIdx = 0, unsigned int _ctxIdx = 0, 
                   const std::vector<unsigned int> _qIdx = { 0 }, unsigned int _pgIdx = 0) : 
            pIdx (_pIdx), dIdx (_dIdx), ctxIdx (_ctxIdx), pgIdx (_pgIdx)
        {
            try
            {
                if (_qIdx.size () != nQueues)
                    throw "The provided vector of command queue indices has the wrong size";

                qIdx = _qIdx;
            }
            catch (const char *error)
            {
                std::cerr << "Error[CLEnvInfo]: " << error << std::endl;
                exit (EXIT_FAILURE);
            }
        }


        /*! \brief Creates a new `CLEnvInfo` object with the specified command queue.
         *  \details Maintains the same OpenCL configuration, but chooses only one
         *           of the available command queues to include.
         *           
         *  \param[in] idx an index for the `qIdx` vector.
         */
        vtkCLEnvInfo<1> GetCLEnvInfo (unsigned int idx)
        {
            try
            {
				std::vector< unsigned int > v;
				v.push_back(qIdx.at( idx ));
                return vtkCLEnvInfo<1> (pIdx, dIdx, ctxIdx, v, pgIdx);
            }
            catch (const std::out_of_range &error)
            {
                std::cerr << "Out of Range error: " << error.what () 
                          << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                exit (EXIT_FAILURE);
            }
        }


        unsigned int pIdx;               /*!< Platform index. */
        unsigned int dIdx;               /*!< Device index. */
        unsigned int ctxIdx;             /*!< Context index. */
        std::vector<unsigned int> qIdx;  /*!< Vector of queue indices. */
        unsigned int pgIdx;              /*!< Program index. */
    };


    /*! \brief A class that collects and manipulates timing information 
     *         about a test.
     *  \details It stores the execution times of a test in a vector, 
     *           and then offers summarizing results.
     *           
     *  \tparam nSize the number of test repetitions.
     *  \tparam rep the type of the values the class stores and returns.
     */
    template <unsigned int nSize, typename rep = double>
    class vtkProfilingInfo
    {
    public:
        /*! \param[in] pLabel a label characterizing the test.
         *  \param[in] pUnit a name for the time unit to be printed 
         *                  when displaying the results.
         */
        vtkProfilingInfo (std::string pLabel = std::string (), std::string pUnit = std::string ("ms")) 
            : label (pLabel), tExec (nSize), tWidth (4 + log10 (nSize)), tUnit (pUnit)
        {
        }

        /*! \param[in] idx subscript index. */
        rep& operator[] (const int idx)
        {
            assert (idx >= 0 && idx < nSize);
            return tExec[idx];
        }

        /*! \brief Returns the sum of the \#nSize executon times.
         *  
         *  \param[in] initVal an initial value from which to start counting.
         *  \return The sum of the vector elements.
         */
        rep total (rep initVal = 0.0)
        {
            return std::accumulate (tExec.begin (), tExec.end (), initVal);
        }

        /*! \brief Returns the mean time of the \#nSize executon times.
         *  
         *  \return The mean of the vector elements.
         */
        rep mean ()
        {
            return total() / (rep) tExec.size ();
        }

        /*! \brief Returns the min time of the \#nSize executon times.
         *  
         *  \return The min of the vector elements.
         */
        rep min ()
        {
            return *std::min_element (tExec.begin (), tExec.end ());
        }

        /*! \brief Returns the max time of the \#nSize executon times.
         *  
         *  \return The max of the vector elements.
         */
        rep max ()
        {
            return *std::max_element (tExec.begin (), tExec.end ());
        }

        /*! \brief Returns the relative performance speedup wrt `refProf`.
         *  
         *  \param[in] refProf a reference test.
         *  \return The factor of execution time decrease.
         */
        rep speedup (vtkProfilingInfo &refProf)
        {
            return refProf.mean () / mean ();
        }

        /*! \brief Displays summarizing results on the test.
         *  
         *  \param[in] title a title for the table of results.
         *  \param[in] bLine a flag for whether or not to print a newline 
         *                   at the end of the table.
         */
        void print (const char *title = nullptr, bool bLine = true)
        {
            std::ios::fmtflags f (std::cout.flags ());
            std::cout << std::fixed << std::setprecision (3);

            if (title)
                std::cout << std::endl << title << std::endl << std::endl;
            else
                std::cout << std::endl;

            std::cout << " " << label << std::endl;
            std::cout << " " << std::string (label.size (), '-') << std::endl;
            std::cout << "   Mean   : " << std::setw (tWidth) << mean ()  << " " << tUnit << std::endl;
            std::cout << "   Min    : " << std::setw (tWidth) << min ()   << " " << tUnit << std::endl;
            std::cout << "   Max    : " << std::setw (tWidth) << max ()   << " " << tUnit << std::endl;
            std::cout << "   Total  : " << std::setw (tWidth) << total () << " " << tUnit << std::endl;
            if (bLine) std::cout << std::endl;

            std::cout.flags (f);
        }

        /*! \brief Displays summarizing results on two tests.
         *  \details Compares the two tests by calculating the speedup 
         *           on the mean execution times.
         *  \note I didn't bother handling the units. It's your responsibility 
         *        to enforce the same unit of time on the two objects.
         *  
         *  \param[in] refProf a reference test.
         *  \param[in] title a title for the table of results.
         */
        void print (vtkProfilingInfo &refProf, const char *title = nullptr)
        {
            if (title)
                std::cout << std::endl << title << std::endl;

            refProf.print (nullptr, false);
            print (nullptr, false);

            std::cout << std::endl << " Benchmark" << std::endl << " ---------" << std::endl;
            
            std::cout << "   Speedup: " << std::setw (tWidth) << speedup (refProf) << std::endl << std::endl;
        }

    private:
        std::string label;  /*!< A label characterizing the test. */
        std::vector<rep> tExec;  /*!< Execution times. */
        uint8_t tWidth;  /*!< Width of the results when printing. */
        std::string tUnit;  /*!< Time unit to display when printing the results. */
    };


    /*! \brief A class for measuring execution times.
     *  \details CPUTimer is an interface for `std::chrono::duration`.
     *  
     *  \tparam rep the type of the value returned by `duration`.
     *  \tparam period the unit of time for the value returned by `duration`.
     *                 It is declared as an `std::ratio<std::intmax_t num, std::intmax_t den>`.
     */
    template <typename rep = int64_t, typename period = std::milli>
    class CPUTimer
    {
    public:
        /*! \brief Constructs a timer.
         *  \details The timer doesn't start automatically.
         * 
         *  \param[in] initVal a value to initialize the timer with.
         */
        CPUTimer (int initVal = 0) : tDuration (initVal)
        {
        }

        /*! \brief Starts the timer.
         *  
         *  \param[in] tReset a flag for resetting the timer before the timer starts. 
         *                    If `false`, the timer starts counting from 
         *                    the point it reached the last time it stopped.
         */
        void start (bool tReset = true)
        {
            if (tReset)
                reset ();

            tReference = std::chrono::high_resolution_clock::now ();
        }

        /*! \brief Stops the timer. 
         *
         *  \return The time measured up to this point in `period` units.
         */
        rep stop ()
        {
            tDuration += std::chrono::duration_cast< std::chrono::duration<rep, period> > 
                (std::chrono::high_resolution_clock::now () - tReference);

            return duration ();
        }

        /*! \brief Returns the time measured by the timer.
         *  \details This time is measured up to the point the timer last time stopped.
         *  
         *  \return The time in `period` units.
         */
        rep duration ()
        {
            return tDuration.count ();
        }

        /*! \brief Resets the timer. */
        void reset ()
        {
            tDuration = std::chrono::duration<rep, period>::zero ();
        }

    private:
        /*! A reference point for when the timer started. */
        std::chrono::time_point<std::chrono::high_resolution_clock> tReference;
        /*! The time measured by the timer. */
        std::chrono::duration<rep, period> tDuration;
    };


    /*! \brief A class for profiling CL devices.
     *
     *  \tparam period the unit of time for the value returned by `duration`.
     *                 It is declared as an `std::ratio<std::intmax_t num, std::intmax_t den>`.
     */
    template <typename period = std::milli>
    class GPUTimer
    {
    public:
        /*! \param[in] device the targeted for profiling CL device.
         */
        GPUTimer (cl::Device &device)
        {
            period tPeriod;
            size_t tRes = device.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION> ();  // x nanoseconds
            // Converts nanoseconds to seconds and then to the requested scale
            tUnit = (double) tPeriod.den / (double) tPeriod.num / 1000000000.0 * tRes;
        }

        /*! \brief Returns a new unpopulated event.
         *  \details The last populated event gets dismissed.
         *  
         *  \return An event for the profiling process.
         */
        cl::Event& event ()
        {
            return pEvent;
        }

        /*! \brief This is an interface for `cl::Event::wait`.
         */
        void wait ()
        {
            pEvent.wait ();
        }

        /*! \brief Returns the time measured by the timer.
         *  \note It's important that it's called after a call to `wait`.
         *
         *  \return The time in `period` units.
         */
        double duration ()
        {
            cl_ulong start = pEvent.getProfilingInfo<CL_PROFILING_COMMAND_START> ();
            cl_ulong end = pEvent.getProfilingInfo<CL_PROFILING_COMMAND_END> ();

            return (end - start) * tUnit;
        }

    private:
        cl::Event pEvent;  /*!< The profiling event. */
        double tUnit;  /*!< A factor to set the scale for the measured time. */
    };

}

#endif  // VTKCLUTILS_HPP
