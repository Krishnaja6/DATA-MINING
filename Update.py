import requests
import json
import redis


url = "https://restcountries.com/v3.1/all"
querystring = {"format": "json", "contains": "C%23", "idRange": "0-150", "blacklistFlags": "nsfw,racist"}
headers = {
    "X-RapidAPI-Key": "2f93a3ff85msh35dd6b619a27a23p117725jsn5337fd510674",
    "X-RapidAPI-Host": "jokeapi-v2.p.rapidapi.com"
}
response = requests.get(url, headers=headers, params=querystring)

if response.status_code == 200:
    data = response.json()

    
    print("Fetched Data:")
    print(json.dumps(data, indent=4))  

   
    r = redis.Redis(
        host='redis-11515.c321.us-east-1-2.ec2.cloud.redislabs.com',
        port=11515,
        password='DTRXwssCMRVIYDSQjdi67P9iib4o5SOu')

    
    key = 'rest_countries_data'
    if r.exists(key):
       
        existing_data = json.loads(r.get(key))
       
        new_country = {"name": "New Country", "capital": "New Capital"}
        existing_data.append(new_country)
        
        r.set(key, json.dumps(existing_data))
        print("Data updated in Redis successfully.")
    else:
        
        r.set(key, json.dumps(data))
        print("Data inserted into Redis successfully.")

else:
    print("Error:", response.text)
