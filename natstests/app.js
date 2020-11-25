const NATS = require('nats')
const nc = NATS.connect({ url: "nats://192.168.0.254:4222", 'preserveBuffers': true })
console.log('connected')
setInterval(() => {
var start = Date.now()
nc.request('frame', '1', { max: 1, timeout: 1000 }, (msg) => {
  if (msg instanceof NATS.NatsError && msg.code === NATS.REQ_TIMEOUT) {
    console.log('request timed out')
  } else {
    console.log(msg)
  }
  console.log('response in ' + (Date.now() - start))
})
}, 2000)
