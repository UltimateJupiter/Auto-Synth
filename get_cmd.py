def base_command(generator, vst, midi, param, flname):
    ret = "{} --channels 1 --quiet --plugin \"{}\" --midi-file {} {} --output \"{}\"".format(generator, vst, midi, param, flname)
    return ret
