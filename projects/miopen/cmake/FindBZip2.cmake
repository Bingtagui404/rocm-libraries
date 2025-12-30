# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[
FindBZip2
---------

Try to find BZip2

IMPORTED Targets
^^^^^^^^^^^^^^^^
This module defines :prop_tgt:`IMPORTED` target ``BZip2::BZip2``, if
BZip2 has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``BZIP2_INCLUDE_DIR``
    the BZip2 include directories
``BZIP2_LIBRARY``
  Link this to use BZip2
``BZIP2_VERSION_STRING``
  the version of BZip2 found

Cache variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``BZIP2_INCLUDE_DIR``
  the BZip2 include directory
#]=======================================================================]

set(_BZIP2_PATH ${PROJECT_SOURCE_DIR}/build)

find_path(BZIP2_INCLUDE_DIR bzlib.h PATHS ${_BZIP2_PATH} PATH_SUFFIXES include)
mark_as_advanced(BZIP2_INCLUDE_DIR)

if(BZIP2_INCLUDE_DIR AND EXISTS "${BZIP2_INCLUDE_DIR}/bzlib.h")
    file(STRINGS "${BZIP2_INCLUDE_DIR}/bzlib.h" BZLIB_H REGEX "bzip2/libbzip2 version [0-9]+\\.[^ ]+ of [0-9]+ ")
    string(REGEX REPLACE ".* bzip2/libbzip2 version ([0-9]+\\.[^ ]+) of [0-9]+ .*" "\\1" BZIP2_VERSION_STRING "${BZLIB_H}")
endif()

add_library(BZip2::BZip2 UNKNOWN IMPORTED)
set_target_properties(BZip2::BZip2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${BZIP2_INCLUDE_DIR}")

find_library(BZIP2_LIBRARY NAMES bz2 bzip2 libbz2 libbzip2 NAMES_PER_DIR PATHS ${_BZIP2_PATH})
if(BZIP2_LIBRARY)
    set_property(TARGET BZip2::BZip2 APPEND PROPERTY IMPORTED_LOCATION "${BZIP2_LIBRARY}")
endif()

