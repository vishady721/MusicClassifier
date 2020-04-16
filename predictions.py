import spotipy
import spotipy.util as util
import sys
import csv

scope = 'user-library-read'

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage: %s username" % (sys.argv[0],))
    sys.exit()

token = util.prompt_for_user_token(username, scope)

if token:
    with open('predictions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        sp = spotipy.Spotify(auth=token)
        results = sp.audio_analysis('1yLSqxA6lYJtPEH0fxKngW')
        for i in range(9, 309):
            writer.writerow([elem for elem in results['segments'][i]['timbre']] + ['2'])
else:
    print("Can't get token for", username)

