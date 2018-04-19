# Troubleshooting guide

This page lists possible issues you might encounter and resolutions for them.

## CUDA GPU backend

The following issues are related to using tensors on CUDA GPUs.

### You get the message `Unable to load DLL 'nvcuda': The specified module could not be found.`

No nVidia GPU driver that supports CUDA is installed.
Install the latest GPU driver from <http://www.nvidia.com/Download/index.aspx>.

### You get the message `The CUBLAS library was not initialized.`

The installed nVidia GPU driver is too old, i.e. its CUDA capability version is too low for the used CUBLAS version.
Install the latest GPU driver from <http://www.nvidia.com/Download/index.aspx>.
