cmake_minimum_required(VERSION 3.10)

file(READ "${CMAKE_CURRENT_LIST_DIR}/NAM/version.h" _nam_version_header)
foreach(_component MAJOR MINOR PATCH)
	string(REGEX MATCH
		"#define[ \t]+NEURAL_AMP_MODELER_DSP_VERSION_${_component}[ \t]+([0-9]+)"
		_nam_version_match "${_nam_version_header}"
	)
	if(NOT _nam_version_match)
		message(FATAL_ERROR "Could not parse NEURAL_AMP_MODELER_DSP_VERSION_${_component} from NAM/version.h")
	endif()
	set(_nam_version_${_component} "${CMAKE_MATCH_1}")
endforeach()

set(_nam_project_version "${_nam_version_MAJOR}.${_nam_version_MINOR}.${_nam_version_PATCH}")
project(NAM VERSION "${_nam_project_version}")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Release, Debug, RelWithDebInfo, MinSizeRel)" FORCE)
	message(STATUS "No build type specified; defaulting to Debug")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use libc++ on macOS, system default on Linux
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

option(NAM_USE_LIBCXX_LINUX "Use libc++ on Linux (Clang + libc++-dev packages; tests SlimmableWavenet _LIBCPP_VERSION path)" OFF)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	include_directories(SYSTEM /usr/local/include)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	if (NAM_USE_LIBCXX_LINUX)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
		# Clang links libc++/libc++abi; do not pull in libstdc++fs (libstdc++).
	else()
		link_libraries(stdc++fs)
	endif()
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
	message(FATAL_ERROR "Unrecognized Platform!")
endif()

# Eigen and AudioDSPTools are NOT vendored here any more: this checkout lives inside the
# AbsoluteStereoCollection monorepo, which owns the single copy of each (submodules/eigen and
# submodules/AudioDSPTools), so the plugins and this library's own tests build against exactly the
# same Eigen. Override these when building the library outside the monorepo layout.
set(NAM_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../eigen" CACHE PATH "Eigen source directory")
set(NAM_AUDIO_DSP_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../AudioDSPTools" CACHE PATH "AudioDSPTools source directory")
foreach(_nam_dep_dir NAM_EIGEN_DIR NAM_AUDIO_DSP_TOOLS_DIR)
	if(NOT EXISTS "${${_nam_dep_dir}}")
		message(FATAL_ERROR "${_nam_dep_dir} does not exist: ${${_nam_dep_dir}}. Run `git submodule update --init` in the monorepo, or pass -D${_nam_dep_dir}=<path>.")
	endif()
endforeach()
# Only the single-header nlohmann/json is still vendored under Dependencies/.
set(NAM_DEPS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies")
include_directories(SYSTEM "${NAM_EIGEN_DIR}")

# Build the specialized A2 fast-path WaveNet (A2-Full + A2-Lite). When ON,
# models whose config matches the A2 shape signature are routed to a hand-optimized
# WaveNet implementation instead of the generic one. When OFF, all models go
# through the generic path.
option(NAM_ENABLE_A2_FAST "Build the A2 fast-path WaveNet" ON)
if(NAM_ENABLE_A2_FAST)
	add_compile_definitions(NAM_ENABLE_A2_FAST)
endif()

add_subdirectory(tools)

#file(MAKE_DIRECTORY build/tools)

#add_custom_target(copy_tools ALL
#	${CMAKE_COMMAND} -E copy "$<TARGET_FILE:tools>" tools/
#	DEPENDS tools
#)
