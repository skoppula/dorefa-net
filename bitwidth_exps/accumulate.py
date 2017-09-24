acts = [32]
weights = list(range(16,34,2))

for act in acts:
    for weight in weights:
        with open('train_logs_a%d_w%d/log.log' % (act, weight), 'r') as f:
            errors = []
            for line in f:
                if 'val-utt-error' in line:
                    errors.append(float(line.split(': ')[-1].strip()))
            min_error = min(errors)
            argmin_error = errors.index(min_error)
            print(act, weight, min_error, argmin_error, len(errors))
