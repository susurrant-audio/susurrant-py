import sys
import json
from collections import defaultdict
from track_info import get_track_comments

def timestamp_to_segment(stamp,
                         sample_rate=22050,
                         frame_size=1024,
                         frames_per_segment=8192):
    seconds = stamp / 1000.0
    sample = seconds * sample_rate
    frame = int(sample / frame_size)
    return frame // frames_per_segment

def parse_comment(comment):
    parsed = {}
    track_id = comment['track_id']
    timestamp = comment.get('timestamp')
    if timestamp is None:
        segment = 0
    else:
        segment = timestamp_to_segment(comment['timestamp'])
    user = comment['user']
    parsed['body'] = comment['body']
    parsed['track_id'] = track_id
    parsed['segment'] = '{}#{:#06x}'.format(track_id, segment)
    parsed['link'] = comment['uri']
    parsed['username'] = user['username']
    parsed['user_url'] = user['permalink_url']
    return parsed

def extract_comments(out_file="../parsed_comments.json"):
    #  track_comments = get_track_comments()
    with open("../comments.json", 'r') as f:
        track_comments = json.load(f)
    parsed_comments = defaultdict(list)
    for comment_id, comment in track_comments.iteritems():
        parsed = parse_comment(comment)
        parsed_comments[parsed['segment']].append(parsed)
    with open(out_file, 'wb') as f:
        json.dump(dict(parsed_comments), f)

if __name__ == '__main__':
    extract_comments(sys.argv[1])
