# nn

A very simple neural network in Odin, for learning: one sigmoid hidden layer, trained by
per-sample backprop. Sizes are compile-time, weights are fixed arrays, and nothing allocates.

```odin
import nn "path/to/nn"

state := rand.create(u64(1))
gen := rand.default_random_generator(&state)

net: nn.Net(2, 3, 1)             // inputs, hidden nodes, outputs
nn.net_init(&net, .Sigmoid, gen) // .Linear for regression
nn.net_train_step(&net, input, target, mask, lr, l2)  // mask: which outputs have a target
out := nn.net_forward(&net, input)
```

- `examples/xor`: the original XOR solver (`./build.sh`)
- `tests`: XOR, linear regression, masked outputs, L2 decay, determinism (`./build.sh test`)

surfi uses it to learn corrections to a surf forecast from logged sessions.
