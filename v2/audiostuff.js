let g_stream = null;
let g_mic_select = null;
let g_context = null;
let g_audio_input = null;
let g_processor = null;

function getStream(constraints) {
  if (!constraints) {
    constraints = { audio: true, video: false };
  }
  return navigator.mediaDevices.getUserMedia(constraints);
}

// 列出所有的录音设备
async function InitializeAudioRecorder() {
  g_mic_select = document.querySelector("#micSelect");

  try {
    window.stream = g_stream = getStream();
    console.log("Got stream");
  } catch(err) {
    console.err("Issue getting mic");
  }

  const device_infos = await navigator.mediaDevices.enumerateDevices();
  console.log(device_infos)

  let mics = [];
  for (let i=0; i<device_infos.length; i++) {
    let di = device_infos[i];
    if (di.kind === 'audioinput') {
      mics.push(di);
      let label = di.label || 'Microphone ' + mics.length;
      console.log('Mic ', label + ' ' + di.deviceId)
      const option = document.createElement('option')
      option.value = di.deviceId;
      option.text = label;
      g_mic_select.appendChild(option);
    }
  }
}

// 选定录音设备
async function SelectRecordDevice(device_id) {
  g_stream = await getStream({
    audio: {
      deviceId: {
        exact: device_id
      }
    },
    video: false
  });
  await SetUpRecording();
}

// 在范例中，如果g_recording为false，就在processor中直接退出
// Create my processor & bind events
async function CreateMyProcessor(ctx, options) {
  const myProcessor = new AudioWorkletNode(ctx, 'myprocessor', options);

  myProcessor.port.onmessage = ((event) => {
    const ms = millis();
    //SoundDataCallbackMyAnalyzer(event.data.buffer, event.data.downsampled, event.data.fft_frames);

    if (event.data.buffer != undefined && event.data.downsampled != undefined) {
      console.log("buffer.length=" + event.data.buffer.length + ", " +
                  "downsampled.length=" + event.data.downsampled.length);
    }
    
    //if (event.data.fft_spectrums) {
    //  event.data.fft_spectrums.forEach((spec) => {
    //    g_fft_vis.AddOneEntry(spec);
    //    g_recorderviz.AddSpectrumIfRecording(spec.slice(0, 200), ms);
    //  });
    //}
    //if (event.data.energy_frames) {
    //  event.data.energy_frames.forEach((en) => {
    //    g_recorderviz.AddEnergyReading(en);
    //  });
    //}
  });
  return myProcessor;
}

// 会创建Script Processor
async function SetUpRecording() {
  g_context = new AudioContext();
  sample_rate = g_context.sampleRate;
  console.log("Sample rate: " + sample_rate);
  g_audio_input = g_context.createMediaStreamSource(g_stream);
  let m = await g_context.audioWorklet.addModule('myprocessor.js');
  g_processor = await CreateMyProcessor(g_context, {
    processorOptions: {
      sampleRate: sample_rate
    }
  });
  g_audio_input.connect(g_processor);
}

window.onload = () => {
  InitializeAudioRecorder();

  document.querySelector("#SelectRecordDevice").addEventListener("pointerdown",
    ()=>{
      SelectRecordDevice("MkD106aGTjX8Hx5TX6h3hL63tk2Cyc5FKWPXDlFIMEE=") 
    }
  );

  document.querySelector("#test2").addEventListener("pointerdown",
    ()=>{
      g_processor.port.postMessage({ recording:false });
    }
  )

  document.querySelector("#test3").addEventListener("pointerdown",
    ()=>{
      g_processor.port.postMessage({ recording:true  });
    }
  )
}
