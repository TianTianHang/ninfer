target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q2_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q2_hadamard.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ternary_a16.cu"
)
