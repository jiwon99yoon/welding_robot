
# Copy robots directory to install path

# Force install robots directory and all contents
install(DIRECTORY robots/
  DESTINATION lib/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*"
)

# Install Python modules (like utils)
install(DIRECTORY utils
  DESTINATION lib/${PROJECT_NAME}
)

