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


## Important Notes
- Only certain layers are supported and the list can be found [here](https://github.com/movidius/ncsdk/releases).
- Using deconvolution is twitchy and does not work with strides if `padding="same"` is used. Use `padding="valid"` for strided deconvolutions.
- The following tests were conducted on this architecture: <br>

Input -> Conv(8, 5x5, 1x1, same, ReLU) -> Conv(16, 5x5, 1x1, same, ReLU) -> Conv(16, 5x5, 2x2, same, ReLU) -> Deconv(X, YxY, ZxZ, valid, None) -> Ouptut <br>

Where Conv(8, 5x5, 1x1, same, ReLU) means  a convolutional layer with 8 filters, 5x5 kernel sized convolutions, 1x1 strides, same padding and ReLU activation.
- 

```
E: [         0] dispatcherEventReceive:236	dispatcherEventReceive() Read failed -4

E: [         0] eventReader:254	Failed to receive event, the device may have reset

E: [         0] ncFifoReadElem:2736	Failed to read fifo element

GetResult exception
[Error 25] Myriad Error: "Status.ERROR".
```

```
[Error 5] Toolkit Error: Stage Details Not Supported: Wrong deconvolution output shape.
```
