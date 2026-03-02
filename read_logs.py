from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

log_path = '/home/kovtun_rs/ReferSeg/mvggt/logs/events.out.tfevents.1771860726.ccmplanner.mipt.ru.3164024.0'
# log_path = '/home/kovtun_rs/ReferSeg/mvggt/logs/events.out.tfevents.1771840942.ccmplanner.mipt.ru.3017115.0'
event_acc = EventAccumulator(log_path).Reload()
scalar_tags = event_acc.Tags()['scalars']

# Retrieve and iterate through data
for tag in scalar_tags:
    print(f"\nData for {tag}:")
    tag_vals = np.array([scalar.value for scalar in event_acc.Scalars(tag)])
    print(f'mean:\t {tag_vals.mean():.3}')
    print(f'median:\t {np.median(tag_vals):.3}')
    print(f'min:\t {tag_vals.min():.3}')
    print(f'max:\t {tag_vals.max():.3}')