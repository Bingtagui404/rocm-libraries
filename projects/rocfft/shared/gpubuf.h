// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCFFT_GPUBUF_H
#define ROCFFT_GPUBUF_H

#include "rocfft_hip.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace
{
    int device_hosting(const void* device_alloc)
    {
        if(!device_alloc)
            throw std::invalid_argument("nullptr device_alloc have no location");
        hipPointerAttribute_t attributes;
        const auto            hip_status = hipPointerGetAttributes(&attributes, device_alloc);
        if(hip_status != hipSuccess)
            throw std::runtime_error("pointer attributes could not be determined (hip_status = "
                                     + std::to_string(hip_status) + ")");
        if(attributes.device == hipInvalidDeviceId)
            throw std::invalid_argument("Invalid device ID fetched from pointer attributes (is it "
                                        "a registered allocation?)");
        return attributes.device;
    }

    //bool alloc_is_accessible_by_device(const void* device_alloc, int device_requesting_access)
    //{
    //    if(device_requesting_access < 0)
    //        throw std::invalid_argument("Invalid device ID requesting access to allocation");
    //    const auto hosting_device = device_hosting(device_alloc);
    //    if(hosting_device == device_requesting_access)
    //        return true;
    //
    //    int        tmp = -1; // must be set to 0,1 upon return
    //    const auto hip_status
    //        = hipDeviceCanAccessPeer(&tmp, device_requesting_access, hosting_device);
    //    if(hip_status != hipSuccess || tmp < 0 || tmp > 1)
    //        throw std::runtime_error("hipDeviceCanAccessPeer failed (hip_status: "
    //                                 + std::to_string(hip_status) + ", tmp: " + std::to_string(tmp)
    //                                 + ")");
    //    return tmp; // implicit conversion to bool
    //}
}

// Simple RAII class for GPU buffers.  T is the type of pointer that
// data() returns
template <class T = void>
class gpubuf_t
{
public:
    gpubuf_t() {}
    // buffers are movable but not copyable
    gpubuf_t(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(owned, other.owned);
        std::swap(bsize, other.bsize);
        std::swap(device, other.device);
        std::swap(is_managed_memory, other.is_managed_memory);
    }
    gpubuf_t& operator=(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(owned, other.owned);
        std::swap(bsize, other.bsize);
        std::swap(device, other.device);
        std::swap(is_managed_memory, other.is_managed_memory);
        return *this;
    }
    gpubuf_t(const gpubuf_t&) = delete;
    gpubuf_t& operator=(const gpubuf_t&) = delete;

    static gpubuf_t make_nonowned(T* p, size_t size_bytes = 0)
    {
        gpubuf_t ret;
        ret.owned             = false;
        ret.buf               = p;
        ret.bsize             = size_bytes;
        ret.is_managed_memory = false; // irrelevant if not owned
        return ret;
    }

    ~gpubuf_t()
    {
        free();
    }

    static bool use_alloc_managed()
    {
        return std::getenv("ROCFFT_MALLOC_MANAGED");
    }

    hipError_t alloc(const size_t size, bool make_it_shared = false)
    {
        // remember the device that was current as of alloc, so we can
        // free on the correct device
        auto ret = hipGetDevice(&device);
        if(ret != hipSuccess)
            return ret;

        bsize             = size;
        is_managed_memory = use_alloc_managed() || make_it_shared;
        free();
        ret = is_managed_memory ? hipMallocManaged(&buf, bsize) : hipMalloc(&buf, bsize);
        if(ret != hipSuccess)
        {
            buf   = nullptr;
            bsize = 0;
        }
        return ret;
    }

    size_t size() const
    {
        return bsize;
    }

    void free()
    {
        if(buf != nullptr)
        {
            if(owned)
            {
                // free on the device we allocated on
                rocfft_scoped_device dev(device);
                (void)hipFree(buf);
            }
            buf   = nullptr;
            bsize = 0;
        }
        owned = true;
    }

    // return a pointer to the allocated memory, offset by the
    // specified number of bytes
    T* data_offset(size_t offset_bytes = 0) const
    {
        void* ptr = static_cast<char*>(buf) + offset_bytes;
        return static_cast<T*>(ptr);
    }

    T* data() const
    {
        return static_cast<T*>(buf);
    }

    // equality/bool tests
    bool operator==(std::nullptr_t n) const
    {
        return buf == n;
    }
    bool operator!=(std::nullptr_t n) const
    {
        return buf != n;
    }
    operator bool() const
    {
        return buf;
    }

private:
    // The GPU buffer
    void* buf = nullptr;
    // whether this object owns the 'buf' pointer (and hence needs to
    // free it)
    bool   owned             = true;
    bool   is_managed_memory = false;
    size_t bsize             = 0;
    int    device            = 0;
};

// default gpubuf that gives out void* pointers
typedef gpubuf_t<> gpubuf;
#endif
