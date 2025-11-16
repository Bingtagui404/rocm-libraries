/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterAllocator.hpp>
#include <rocRoller/Utilities/Error.hpp>

// Used for std::iota
#include <numeric>

namespace rocRoller
{
    namespace Register
    {
        template <std::ranges::forward_range T>
        AllocationPtr Allocator::reassign(T const& indices)
        {
            std::vector<int> registers(indices.begin(), indices.end());

            AssertFatal(registers.size() != 0);

            auto firstAlloc = m_registers.at(registers[0]).lock();
            AssertFatal(firstAlloc);

            for(auto index : registers)
                AssertFatal(m_registers.at(index).lock() == firstAlloc);

            auto newAlloc = std::make_shared<Allocation>(
                firstAlloc->m_context.lock(),
                m_regType,
                DataType::Raw32,
                registers.size(),
                Register::AllocationOptions{.contiguousChunkWidth = Register::MANUAL});

            allocate(newAlloc, registers);

            return newAlloc;
        }
    }
}
