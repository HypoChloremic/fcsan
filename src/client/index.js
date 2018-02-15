

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