find_path(MATIO_INCLUDE_DIR NAMES matio.h 
                            PATHS ${MATIO_ROOT_DIR} ${MATIO_ROOT_DIR}/include)

find_library(MATIO_LIBRARIES NAMES matio
                             PATHS ${MATIO_ROOT_DIR} ${MATIO_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MATIO DEFAULT_MSG MATIO_INCLUDE_DIR MATIO_LIBRARIES)

if(MATIO_FOUND)
  message(STATUS "Found matio    (include: ${MATIO_INCLUDE_DIR}, library: ${MATIO_LIBRARIES})")
  mark_as_advanced(MATIO_INCLUDE_DIR MATIO_LIBRARIES)

  caffe_parse_header(${MATIO_INCLUDE_DIR}/matio_pubconf.h
                     MATIO_VERSION_LINES MATIO_MAJOR_VERSION MATIO_MINOR_VERSION MATIO_RELEASE_LEVEL)
  set(MATIO_VERSION "${MATIO_MAJOR_VERSION}.${MATIO_MINOR_VERSION}.${MATIO_RELEASE_LEVEL}")
endif()
