var root = document.body;

var Splash = {
  view: function() {
    return m("a", {href:"#!/hello"}, "Enter!")
  }
}

var Hello = {
  count: 0,
  state: "Not clicked",
  view: function() {
    return m("main",[
      m("h1", {class:"title"}, "My first app"),
      m("div", ""+this.state),
      m("button", {
        onmousedown: ()=>{
          this.count++;
          this.state = "Clicked";
        },
        onmouseup: ()=>{
          this.state = "Not clicked";
        },
      }, this.count+" clicks"),
      m("a", {href:"#!/splash"}, "Return"),
    ])
  }
}

m.route(root, "/splash", {
  "/splash": Splash,
  "/hello": Hello,
})