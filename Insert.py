import requests
import json
import redis


url = "https://restcountries.com/v3.1/all"
querystring = {"format": "json"}
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

    
    r.set('rest_countries_data', json.dumps(data))

    print("\nData inserted into Redis successfully.")

    
    search_country = "India" 
    found_countries = [country for country in data if search_country in country.get('name', '')]

    if found_countries:
        print(f"\nFound {len(found_countries)} countries matching '{search_country}':")
        for country in found_countries:
            print(json.dumps(country, indent=4)) 
    else:
        print(f"\nNo countries found matching '{search_country}'.")
else:
    print("Error:", response.text)
