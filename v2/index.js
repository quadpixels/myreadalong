var root = document.body;
var g_canvas = document.createElement("canvas", "g_canvas");
var g_myworker_wrapper = new MyWorkerWrapper();

var g_aligner = new Aligner();
g_aligner.LoadSampleData();

var Splash = {
  view: function() {
    return m("a", {href:"#!/hello"}, "Enter!")
  }
}

function GetAudioContextState(x) {
  let ret = "Sample rate: " + x.sampleRate + " Hz<br/>";
  ret = ret + "currentTime: " + x.currentTime + "<br/>";
  return ret;
}

var Hello = {
  count: 0,
  frame_count: 0,
  state: "Not clicked",
  ctx: g_canvas.getContext('2d'),
  log_entries: [],
  num_in_flight: 0,
  views: function() {
    ret = [
      m("div", ""+this.state),
      m("br"),
      m("div", {
        style: {
          width: '100%',
          height: '32px',
          border: '1px blue solid',
        },
        onmousedown: () => {
          //this.state = "mousedown";
          //this.OnMouseDown(); 
        },
        onmouseup: () => { 
          //this.state = "mouseup"
          //this.OnMouseUp(); 
        },
        ontouchdown: () => {
          this.state = "touchdown";
          this.OnMouseDown(); 
        },
        ontouchup: () => { 
          this.state = "touchup";
          this.OnMouseUp(); 
        }
      }, this.count+" clicks"),
      m("div", "frame " + this.frame_count),
      m("div", "g_record_buffer_orig.length=" + g_record_buffer_orig.length),
      m("div", "g_record_buffer_16khz.length=" + g_record_buffer_16khz.length),
      m("div", "g_fft_buffer.length=" + g_fft_buffer.length),
      m("div", this.num_in_flight + " in-flight reqs"),
      m("input", {
        type: "submit",
        value: this.num_in_flight > 0 ? "等待识别("+ this.num_in_flight + ")" : "按键录音",
        "onpointerdown": ()=>{ Hello.OnMouseDown(); },
        "onpointerup": ()=>{ Hello.OnMouseUp(); },
        disabled: this.num_in_flight > 0,
      })
    ];

    const num_shown = 20;
    for (let i=0; i<num_shown; i++) {
      let idx = this.log_entries.length - num_shown + i;
      if (idx >= 0 && idx < this.log_entries.length) {
        ret.push(m("div", {}, this.log_entries[idx]));
      }
    };

    return ret;
  },
  view: function() {
    const w = g_canvas.width, h = g_canvas.height;
    this.ctx = g_canvas.getContext('2d')
    this.ctx.fillStyle = '#0000FF'
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.font = "regular 12px sans-serif"
    this.ctx.textBaseline = "top";
    this.ctx.fillText("count=" + this.count, 3, 2);
    this.ctx.fillText("frame " + this.frame_count, 3, 12);
    let c = "g_context is null";
    if (g_context != null) {
      c = "g_context.state=" + g_context.state;
    }

    c = "g_record_buffer_orig.length=" + g_record_buffer_orig.length;
    this.ctx.fillText(c, 3, 22);

    c = "g_record_buffer_16khz.length=" + g_record_buffer_16khz.length;
    this.ctx.fillText(c, 3, 32);

    c = "g_fft_buffer.length=" + g_fft_buffer.length;
    this.ctx.fillText(c, 3, 42);

    c = this.num_in_flight + " in-flight reqs";
    this.ctx.fillText(c, 3, 52);

    this.ctx.strokeStyle = '#0000FF';

    this.ctx.beginPath();
    this.ctx.moveTo(0, 0);
    this.ctx.lineTo(w, h);
    this.ctx.stroke();

    let txt = ""
    /*
    if (g_audio_input == null) {
      txt = "g_audio_input is null";
    } else {
      txt = "g_audio_input.channelCount=" + g_audio_input.channelCount.toString();
    }
    this.ctx.fillText(txt, 3, 22);*/

    return m("main",
      this.views()
    )
  },
  OnMouseDown: function() {
    this.count++
    g_aligner.OnStartRecording();
    g_processor.port.postMessage({ recording: true });
  },
  OnMouseUp: function() {
    g_processor.port.postMessage({ recording: false });
  },
  AddLogEntry: function(entry) {
    this.log_entries.push(entry);
  }
}

m.mount(document.body,
  {
    view:() => [
      m(Hello),
      m(g_aligner),
    ]
  }
)

let g_prev_timestamp;
function step(timestamp) {
  if (true) {
    Hello.frame_count ++;
    m.redraw();
  }
  window.requestAnimationFrame(step);
}
window.requestAnimationFrame(step);

document.body.appendChild(g_canvas)

// ======================================

window.onload = async () => {
  Hello.AddLogEntry("window.onload");
  await InitializeAudioRecorder();
  g_myworker_wrapper.InitializeMyWorker();
  SelectRecordDevice(document.querySelector("#micSelect").value) 

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

      audio_url = WriteWAVFileToBlob(g_record_buffer_16khz, 16000);
      a = document.querySelector("#audio_16khz");
      d = document.querySelector("#download_16khz");
      a.setAttribute("src", audio_url);
      d.setAttribute("href", audio_url);
      d.download = "output_16khz.wav";
    }
  );

  document.querySelector("#ClearButton").addEventListener("pointerdown",
    ()=>{
      g_record_buffer_16khz = [];
      g_record_buffer_orig = [];
      g_fft_buffer = [];
      g_aligner.Clear();
    }
  );

  document.querySelector("#LoadTfjs").addEventListener("pointerdown",
    ()=>{
      g_myworker_wrapper.InitializeMyWorker();
    }
  )

  document.querySelector("#DoPredictButton").addEventListener("pointerdown",
    ()=>{
      // 把所有当前存着的FFT都丢进预测器
      const window_width = 100;
      const window_delta = 25;
      for (let i=0; i<g_fft_buffer.length; i+=window_delta) {
        let ffts = g_fft_buffer.slice(i, i+window_width);
        ffts = PadZero(ffts, window_width);
        let ts = i * (1.0 / 16000);
        g_myworker_wrapper.Predict(ts, i, ffts);
      }
    }
  )
}

function PadZero(ffts, len) {
  while (ffts.length < len) {
    let zeroes = [];
    for (let i=0; i<200; i++) { zeroes.push(0); }
    ffts.push(zeroes);
  }
  return ffts;
}

window.addEventListener('keydown', function(event) {
  if (event.keyCode == 37) {  // left arrow
    g_aligner.do_PrevStep();
  } else if (event.keyCode == 39) { // Right arrow
    g_aligner.do_NextStep();
  }
})