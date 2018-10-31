# prg_prgncs
PRG's Setup of Intel Neural Compute Stick

## Setup Instructions
- Start afresh with new Ubuntu 16.04 as the NCS API requires its own version of OpenCV which conflicts with other OpenCV Installations
- Install Movidis SDK 2 from https://github.com/movidius/ncsdk. Basic Instructions are given below:
  - ```
    git clone -b ncsdk2 https://github.com/movidius/ncsdk.git
    ```
  - ```
    make install
    ```
    

## TODO
- [ ] Test CIFAR10 Custom Example on NCS and benchmark speed
- [ ] Test CIFAR10 Custom Example with Deconv in-between and benchmark speed
- [ ] Test CIFAR10 Custom Example to learn Identity transform Input->Conv->Deconv 
