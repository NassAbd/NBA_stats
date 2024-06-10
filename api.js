const http = require('https');

const options = {
	method: 'GET',
	hostname: 'api-nba-v1.p.rapidapi.com',
	port: null,
	path: '/teams',
	headers: {
		'x-rapidapi-key': 'fe859d29c5msh1bdcdf8e30c6c19p165c45jsn539e1b3cfd5b',
		'x-rapidapi-host': 'api-nba-v1.p.rapidapi.com'
	}
};

const req = http.request(options, function (res) {
	const chunks = [];

	res.on('data', function (chunk) {
		chunks.push(chunk);
	});

	res.on('end', function () {
		const body = Buffer.concat(chunks);
		console.log(body.toString());
	});
});

req.end();