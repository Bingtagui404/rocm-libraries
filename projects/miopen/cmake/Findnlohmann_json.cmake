# Create imported target nlohmann_json::nlohmann_json
add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)

set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
)