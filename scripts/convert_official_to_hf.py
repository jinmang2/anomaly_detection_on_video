from collections import OrderedDict


def convert(state_dict: OrderedDict) -> OrderedDict:
    new_state_dict = OrderedDict()

    for key, tensor in state_dict.items():
        if "to_tokens" in key or "to_mag" in key:
            new_state_dict["backbone.amplifier." + key] = tensor
        elif "to_logits" in key:
            new_state_dict["layer_norm." + key.split(".")[-1]] = tensor
        elif "fc" in key:
            new_state_dict[key] = tensor
        elif "stages" in key:
            info = key.split(".")[1:]
            layer = info[0]
            prefix = f"backbone.layers.{layer}."
            # intermediate
            if info[1] == "1":
                layer_name = "layer_norm" if info[2] == "0" else "conv"
                new_state_dict[prefix + f"3.{layer_name}.{info[-1]}"] = tensor
            # blocks
            else:
                depth = info[3]
                prefix += f"{depth}."
                # shortcut convolution
                if info[4] == "0":
                    new_state_dict[prefix + f"scc.{info[-1]}"] = tensor
                # block
                elif info[4] == "1":
                    new_state_dict[prefix + f"attention.{info[-2]}.{info[-1]}"] = tensor
                # ffn
                elif info[4] == "2":
                    if info[-2] == "0":
                        layer_name = "layer_norm"
                    elif info[-2] == "1":
                        layer_name = "in_conv"
                    elif info[-2] == "4":
                        layer_name = "out_conv"
                    else:
                        continue
                    new_state_dict[prefix + f"ffn.{layer_name}.{info[-1]}"] = tensor

    return new_state_dict
