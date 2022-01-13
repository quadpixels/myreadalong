class IntroScreen {
  constructor() {
    this.countdown = 0;
    this.countdown0 = 1000;
    // 门开的程度，1为全开
    this.k0 = 0;
    this.k1 = 0;
    this.k = 0;
    this.visible = true;
    this.state = "";
    this.message = "";
  }

  Update(delta_ms) {
    if (this.countdown >= 0) {
      this.countdown -= delta_ms;
      if (this.countdown < 0) {
        this.countdown = 0;
        if (this.state == "fadeout") {
          this.visible = false;
          this.message = "";
        }
      }
    }
  }

  GetCompletion() {
    return 1 - this.countdown / this.countdown0;
  }

  // 开的程度，1为全开
  GetK() {
    return lerp(this.k0, this.k1, this.GetCompletion());
  }

  SetMessage(x) {
    this.message = x;
  }

  FadeOut() {
    this.visible = true;
    this.k0 = 0;
    this.k1 = 1;
    this.countdown = 1000;
    this.state = "fadeout";
  }

  Render() {
    if (!this.visible) return;
    push();
    noStroke();
    //stroke(color(167, 119, 37));

    fill(color(250, 240, 228));
    let x0 = 0, x1 = W0/2 * (1-this.GetK());
    rect(x0, 0, x1-x0, H0);
    x0 = W0/2 + W0/2 * this.GetK(); x1 = W0;
    rect(x0, 0, x1-x0, H0);

    const a = 255 * (1 - this.GetK());
    stroke(color(167, 119, 37));
    fill(color(213, 173, 114));
    const w1 = W0 / 5 * (1 - this.GetK()), y0 = H0 * 0.15, y1 = H0 * 0.58, y2 = H0 * 0.75;
    const pad_y = 68;
    rect(W0/2-w1/2, y0, w1, y1-y0);

    stroke(43, 20, 0, a);
    fill(169, 58, 39, a)
    const title = ["芝", "麻", "开", "门"];
    textAlign(CENTER, CENTER);
    textSize(48);
    for (let i=0; i<title.length; i++) {
      text(title[i], W0/2, lerp(y0+pad_y, y1-pad_y, i/(title.length-1)));
    }

    textSize(24);
    noStroke();
    fill(213, 173, 114, a);
    text(this.message, W0/2, y2);

    //
    fill(64);
    //text(x0+","+x1+","+this.GetK(), 8, 8)

    pop();
  }
}