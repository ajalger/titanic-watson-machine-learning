"""
Example code for an article on Compose using IBM Watson Machine Learning to predict Titanic survivors. 
https://compose.com/articles/better-decision-making-with-watson-machine-learning-and-compose/
"""

import urllib3, requests, json
import psycopg2 as pg
import csv

compose = {
  "host":"xxxxxx",
  "port":"xxxxxx",
  "user":"xxxxxx",
  "password":"xxxxxx",
  "database":"xxxxxx"
}

conn_string = "host={host} port={port} dbname={database} user={user} password={password}".format(host=compose["host"], port=compose["port"], database=compose["database"], user=compose["user"], password=compose["password"])
conn = pg.connect(conn_string)

wml_credentials = {
    "access_key": "xxxxxx",
    "instance_id": "xxxxxx",
    "password": "xxxxxx",
    "url": "https://ibm-watson-ml.mybluemix.net",
    "username": "xxxxxx"
}

headers = urllib3.util.make_headers(basic_auth="{username}:{password}".format(username=wml_credentials["username"], password=wml_credentials["password"]))
url = "{}/v3/identity/token".format(wml_credentials["url"])
response = requests.get(url, headers=headers)
mltoken = json.loads(response.text).get("token")

header = {"Content-Type": "application/json", "Authorization": "Bearer " + mltoken}

scoring_url = "xxxxxxxxxxxxx"

cursor = conn.cursor()

cursor.execute("SELECT * FROM passenger_prediction")    
results = cursor.fetchall()

def get_passenger_payload(results):
    passengers = list()
    
    for result in results:
        passengers.append(list(result))

    for passenger in passengers:
        if passenger[2]:
            passenger[2] = ""
        if passenger[4] is None:
            passenger[4] = 0
        if passenger[8] is None:
            passenger[8] = 0
        if passenger[9] is None:
            passenger[9] = ""

    return {"fields": ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], "values": passengers}

full_payload = get_passenger_payload(results)

def run_ml_titanic(payload):
    result_scores = requests.post(scoring_url, json=payload, headers=header)
    return result_scores

passenger_scores = run_ml_titanic(full_payload)

def parse_results(scores):
    ml_results = json.loads(scores.text)
    passenger_survival = list()

    for value in ml_results['values']:
        survived = 0
        passenger_id = value[0]
        if value[13][0] < value[13][1]:
            survived = 1
        passenger_survival.append((passenger_id, survived))

    return passenger_survival

survival = parse_results(passenger_scores)
print(survival)
with open("/path/to/newfile.csv", "w") as csvfile:
    write_file = csv.writer(csvfile)
    write_file.writerow(['PassengerId', 'Survived'])
    for passenger in survival:
        p_id = passenger[0]
        prediction = passenger[1]
        write_file.writerow([p_id, prediction])
csvfile.close()

cursor.execute("""UPDATE passenger_prediction SET survived = s.survived FROM unnest(%s) s(p_id int, survived int) where passenger_prediction.passenger_id = s.p_id""", (survival,))
conn.commit()

cursor.close()
