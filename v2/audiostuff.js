// Web Audio API stuff
let g_stream = null;
let g_mic_select = null;
let g_context = null;
var g_audio_input = null;
let g_processor = null;

let g_record_buffer_orig = [];  // 混缩成一个声道的样本
let g_record_buffer_8khz = [];  // 混缩并减采的样本

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
      g_record_buffer_orig = g_record_buffer_orig.concat(Array.from(event.data.buffer));
      g_record_buffer_8khz = g_record_buffer_8khz.concat(event.data.downsampled);
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
  console.log("Audio input channel count: " + g_audio_input.channelCount);
  let m = await g_context.audioWorklet.addModule('myprocessor.js');
  g_processor = await CreateMyProcessor(g_context, {
    processorOptions: {
      sampleRate: sample_rate
    }
  });
  g_audio_input.connect(g_processor);
}

// Copied from CodePen
function writeUTFBytes(view, offset, string){ 
  let lng = string.length;
  for (let i = 0; i < lng; i++){
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function WriteWAVFileToBlob(buf, sample_rate) {
  let buffer = new ArrayBuffer(44 + buf.length * 2);  // 16 bit
  let view = new DataView(buffer);

  writeUTFBytes(view, 0, 'RIFF');  // ChunkID
  view.setUint32(4, 44 + buf.length * 2, true);  // ChunkSize
  writeUTFBytes(view, 8, 'WAVE');  // Format

  writeUTFBytes(view, 12, 'fmt ');  // Subchunk1ID
  view.setUint32(16, 16, true);  // Subchunk1Size
  view.setUint16(20, 1, true);   // AudioFormat
  view.setUint16(22, 1, true);   // NumChannels
  view.setUint32(24, sample_rate, true);      // SampleRate
  view.setUint32(28, sample_rate * 2, true);  // ByteRate
  view.setUint16(32, 4,  true);   // BlockAlign
  view.setUint16(34, 16, true);   // BitsPerSample

  writeUTFBytes(view, 36, 'data');
  view.setUint32(40, buf.length * 2, true);
  for (let i=0; i<buf.length; i++) {
    view.setInt16(44+i*2, buf[i]*(0x7FFF), true);
  }

  const blob = new Blob([view], {type:'audio/wav'});
  const audio_url = URL.createObjectURL(blob);
  return audio_url;
}

window.onload = () => {
  InitializeAudioRecorder();

  document.querySelector("#SelectRecordDevice").addEventListener("pointerdown",
    ()=>{
      SelectRecordDevice(document.querySelector("#micSelect").value) 
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

  document.querySelector("#test4").addEventListener("pointerdown",
    ()=>{
      let audio_url = WriteWAVFileToBlob(g_record_buffer_orig, g_context.sampleRate);
      let a = document.querySelector("#audio_orig");
      let d = document.querySelector("#download_orig");
      a.setAttribute("src", audio_url);
      d.setAttribute("href", audio_url);
      d.download = "output_orig.wav";

      audio_url = WriteWAVFileToBlob(g_record_buffer_8khz, 8000);
      a = document.querySelector("#audio_8khz");
      d = document.querySelector("#download_8khz");
      a.setAttribute("src", audio_url);
      d.setAttribute("href", audio_url);
      d.download = "output_8khz.wav";
    }
  )
}
