# XodDDSPVocode
Audio vocoder with DDSP neural net phase processing 

XodDDSPVox Project Plan

1. Project Overview

Goal: Build a VST3 audio plugin that performs real‑time phase‑vocoder processing using a neural network between STFT and iSTFT.

Components: • ML training pipeline (PyTorch) • Model export to ONNX → TensorRT • C++ inference wrapper and real‑time integration • JUCE VST3 plugin hosting STFT ↔ NN ↔ ISTFT chain



2. Outline Steps

1. Data Preparation

Collect or record 48 kHz stereo WAVs

Generate paired targets (stretch, pitch, style) via SoX or Rubber Band

Organize into training/validation sets



2. Model Architecture & Training

Define conditional U‑Net (encoder/decoder, FiLM layers for control vectors)

Inputs: mag_x (freq×time), control vector c

Output: complex phase map (real+imag)

Losses: waveform L1/L2, multi‑resolution STFT, phase consistency

Optimizer: AdamW, lr 1e‑4…1e‑3, mixed precision



3. Export & Engine Build

Export PyTorch model → ONNX (dynamic time axis)

Build TensorRT engine (trtexec or C++ API) • FP16 mode, shape profiles



4. C++ Inference Integration

Implement TensorRTEngine class: • Load .trt file via TensorRT API • Allocate device buffers for inputs (mag, c) and output (phase) • infer(mag, c) → vector<complex<float>>

Optimize for zero allocations



5. Plugin Integration (JUCE VST3)

In prepareToPlay(): • Initialize FFTW plans, window, ring buffers • Instantiate TensorRTEngine

In processBlock(): • Ring‑buffer input audio frames • On each FFT hop: – Apply window + FFT → mag – Call infer(mag, controlVector) – Rebuild complex bins, IFFT + overlap‑add • Output processed audio

Expose GUI parameters for control vector c





3. Required Tools

Python: PyTorch, torchaudio, sox CLI (for target generation)

ONNX: torch.onnx

TensorRT SDK: trtexec, C++ headers/libs

C++ Frameworks: JUCE 7+, FFTW3 (or KissFFT), CUDA toolkit



4. Important Techniques

STFT/iSTFT: high‑quality overlap‑add, Hann windows

U‑Net with FiLM: conditional feature‑wise modulation

Loss Functions: waveform L1/L2, spectral convergence, multi‑res STFT, phase consistency

Mixed Precision: torch.cuda.amp, TensorRT FP16

Dynamic Shapes: ONNX dynamic time axis, TensorRT shape profiles

Real‑Time Considerations: zero allocations, batch frames vs. latency, GPU vs. audio‑thread offload



5. Key C++ Classes & Components

TensorRTEngine • Methods: loadEngine(path), infer(mag, controlVec) • Manages: IRuntime, ICudaEngine, IExecutionContext, device buffers

PhaseVocoderModel (optional libtorch/ONNX fallback) • TorchScript or ONNX Runtime loader for prototyping

RingBuffer<T> • Circular buffer for streaming audio & overlap‑add

PluginProcessor (inherits AudioProcessor) • Members: TensorRTEngine phaseEngine; Vector<float> window, fftIn; Vector<complex> fftOut; RingBuffer<float> inBuf, outBuf; AudioParameterFloat controls… • Methods: prepareToPlay(), processBlock() implementing STFT↔NN↔iSTFT



6. Next Steps

Write training scripts and prepare dataset

Prototype network in PyTorch; verify audio quality

Export to ONNX, build TensorRT engine, test inference

Scaffold JUCE plugin and integrate inference loop

Profile latency & optimize FFT/inference balance




