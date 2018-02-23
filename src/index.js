console.log("Opened")

// Hoisting is when the declaration of all variables 
// are automatically moved to the top of the function
// note that declaration refers to "let" or "var". 
// initializing at the same time as using var or let does
// not send it up to the top, eg "var y = 10;" instead of 
// simply declaring the shit "var y;"
function SEND(){
	console.log("Running SEND");

	var connection = new WebSocket("ws://127.0.0.1:3000");
	console.log("Connected..");

	// This is like a lambda function that doesnt run upon declaration..
	connection.onopen = function(){
		console.log("Opening, pinging")
		connection.send("1231412")
	};

	connection.onmessage = function(e){
		console.log("Server: " + e.data);
	};

}

function GET(){
	console.log("Running Get")
	brush = document.getElementsByClassName("extent linkedbrush")[0]
	brushWidth  = brush.width.animVal.value;
	brushHeight = brush.height.animVal.value;
	transBrushX = parseFloat(brush.attributes.x.nodeValue);
	transBrushY = parseFloat(brush.attributes.y.nodeValue);

	transX = document.getElementsByClassName("mpld3-path")[0].attributes.transform.nodeValue.split("translate")[1].replace("(","").replace(")","").split(",")[0];
	transY = document.getElementsByClassName("mpld3-path")[0].attributes.transform.nodeValue.split("translate")[1].replace("(","").replace(")","").split(",")[1];
	transX = parseFloat(transX);
	transY = parseFloat(transY);
	
	ptX    = document.getElementsByClassName("mpld3-path")[0].__data__[0];
	ptY    = document.getElementsByClassName("mpld3-path")[0].__data__[1];

	xRatio = ptX/transX;
	yRatio = ptY/transY;

	x1 = xRatio * transBrushX;
	y1 = yRatio * transBrushY;
	
	x2 = x1+(brushWidth*transBrushX);
	y2 = y1+(brushHeight*transBrushY);



	console.log(y1,x1)
	
}