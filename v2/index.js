var root = document.body;
var g_canvas = document.createElement("canvas", "g_canvas");

let g_millis0 = 0;
function millis() {
  if (g_millis0 == 0) {
    g_millis0 = new Date().getTime();
  }
  return new Date().getTime() - g_millis0;
}

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
  views: function() {
    return [
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
//      m("a", {href:"#!/splash"}, "Return"),
    ];
  },
  view: function() {
    const w = g_canvas.width, h = g_canvas.height;
    this.ctx = g_canvas.getContext('2d')
    this.ctx.fillStyle = '#0000FF'
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.font = "regular 12px sans-serif"
    this.ctx.fillText("count=" + this.count, w/2, h/2 - 10);
    this.ctx.fillText("frame " + this.frame_count, w/2, h/2+10);

    return m("main",
      this.views()
    )
  },
  OnMouseDown: function() {
    this.count++;
  },
  OnMouseUp: function() {
  }
}

m.route(root, "/hello", {
  "/splash": Splash,
  "/hello": Hello,
})

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