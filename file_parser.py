import json


'''def read_json_singlefile(filename):
	hyps = []
	refs = []

	with open(filename, 'r') as hyps_f:
		data = json.load(hyps_f)
		hyps = [pred['prediction'] for pred in data]
	with open(filename, 'r') as refs_f:
		data = json.load(refs_f) 
		refs = [ref['reference'] for ref in data]
	return hyps, refs'''

def read_json_singlefile(filename):

    hyps = []
    refs = []

    with open(filename, 'r') as hyps_f:
        for line in hyps_f:
            data = json.loads(line)
            hyps.append(data['generated_code'])
            refs.append(data['reference'])
    return hyps, refs