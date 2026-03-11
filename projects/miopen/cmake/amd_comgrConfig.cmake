# Find amd_comgr

find_path(AMD_COMGR_INCLUDE_DIR amd_comgr.h PATHS ${PROJECT_SOURCE_DIR}/build PATH_SUFFIXES include/amd_comgr)
mark_as_advanced(AMD_COMGR_INCLUDE_DIR)

if(AMD_COMGR_INCLUDE_DIR)
    set(amd_comgr_DIR ${AMD_COMGR_INCLUDE_DIR})
    file(STRINGS ${AMD_COMGR_INCLUDE_DIR}/amd_comgr.h _ver_line
         REGEX "^#define AMD_COMGR_INTERFACE_VERSION_MAJOR *[0-9]"
         LIMIT_COUNT 1)
    string(REGEX MATCH "[0-9]" amd_comgr_VERSION_MAJOR "${_ver_line}")
    unset(_ver_line)
    file(STRINGS ${AMD_COMGR_INCLUDE_DIR}/amd_comgr.h _ver_line
         REGEX "^#define AMD_COMGR_INTERFACE_VERSION_MINOR *[0-9]"
         LIMIT_COUNT 1)
    string(REGEX MATCH "[0-9]" amd_comgr_VERSION_MINOR "${_ver_line}")
    unset(_ver_line)
    set(amd_comgr_VERSION "${amd_comgr_VERSION_MAJOR}.${amd_comgr_VERSION_MINOR}")
    set(amd_comgr_LIBRARY amd_comgr_${amd_comgr_VERSION_MAJOR})
endif()


