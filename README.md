////////////////////////////////////////////////////////////////////////////////
// __Project ::(Z444-2):: XodDDSPVox                                         \\\
////////////////////////////////////////////////////////////////////////////////

Title:
XodDDSPVox – Real-Time ML Phase Vocoder Plugin

Definition:
A VST3 plugin for real-time spectral transformation using a neural network phase processor between STFT and iSTFT.
→ 1st desired outcome: Real-time VST3 plugin with GPU-accelerated phase vocoding
→ 2nd desired outcome: Neural control of pitch, stretch, and style via control vector

// *------------------------------------------------------*
// ((Step 1::Goals))
// *------------------------------------------------------*
* Implement efficient C++ STFT → ML Inference → iSTFT audio pipeline
* Train a U-Net + FiLM model on audio STFT magnitudes to predict complex phase
* Export PyTorch model to ONNX with dynamic time dimension
* Build and deploy TensorRT engine for real-time GPU inference
* Integrate ML inference into JUCE `processBlock()` loop with low latency

// *------------------------------------------------------*
// ((Step 2::Diagnose Root Problems & Eliminate))
// *------------------------------------------------------*
ROOT PROBLEM (1): No valid GPU inference interface in plugin  
SOLUTION (1): TensorRT deployment via ONNX, dynamic axes, FP16 engine

ROOT PROBLEM (2): Format mismatch between C++ STFT and ML input  
SOLUTION (2): Validate tensor shape, scaling, layout (NCHW, batch=1), and control vector

// *------------------------------------------------------*
// ((Step 3::Align & Execute))
// *------------------------------------------------------*
KINGPIN:  
**Define and validate the correct tensor format (STFT mag + control → phase map) used between C++ and ML model.**

This is the most immediate bottleneck. Without a verified tensor format, model training cannot begin correctly, and real-time inference cannot be tested or integrated. This includes defining the shape, scaling, layout (e.g., NCHW or NHWC), dynamic time support, stereo handling, and control vector injection. A small test loop using dummy data and a mock Python model must be implemented to validate C++ ↔ ML format compatibility. Once solved, model training, export, and inference integration can proceed confidently.

// *------------------------------------------------------*
// Links:
// *------------------------------------------------------*
* PyTorch U-Net Dev: ~/ml/models/unet_film.py  
* Dataset: ~/ml/data/stems/  
* TensorRT Docs: https://docs.nvidia.com/deeplearning/tensorrt  
* JUCE Framework: https://juce.com/

// *------------------------------------------------------*
// Notes:
// *------------------------------------------------------*
* Prior project-wide kingpin (goal): Export ONNX with dynamic time axis → build usable TensorRT engine  
* Current blocker is upstream: tensor interface must be solid before export is meaningful  
* Consider audio buffer size vs FFT hop alignment when testing dynamic time dimension

// *------------------------------------------------------*
// Progress:
// *------------------------------------------------------*
start 2025/07/01  
current 2025/07/10  

Freeze State:  
→ STFT/iSTFT and COLA working for stereo input  
→ Control vector design in progress  
→ Next step: verify tensor interface and mock inference cycle

// *------------------------------------------------------*
// Completed Form:
// *------------------------------------------------------*
* C++ plugin with STFT → TensorRT inference → iSTFT
* ML model trained with control vector for pitch/stretch
* Real-time spectral morphing with user-controllable parameters

////////////////////////////////////////////////////////////////////////////////
