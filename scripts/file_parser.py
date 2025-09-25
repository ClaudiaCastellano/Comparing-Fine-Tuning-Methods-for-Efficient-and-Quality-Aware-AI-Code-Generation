import json


def read_json_singlefile(filename):

    hyps = []
    refs = []

    with open(filename, 'r') as hyps_f:
        for line in hyps_f:
            data = json.loads(line)
            hyps.append(data['generated_code'])
            refs.append(data['reference'])
    return hyps, refs