target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q2_a16_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q2_hadamard.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../ptq1/ptq1_linear.cu"
)
