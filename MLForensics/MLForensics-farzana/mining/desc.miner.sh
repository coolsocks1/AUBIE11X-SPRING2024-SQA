#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
  repo_name="https://api.github.com/repos/"$line
  echo "----------------------------------------------------------------------------"
  echo $repo_name 
  jsonFileName=`echo $line | tr '/' _`
  echo $jsonFileName
  curl -H "Authorization: token <TOKEN_HERE>" -H "Accept: json" -ni $repo_name > $jsonFileName.json
  cat $jsonFileName".json" | grep -w '"description": '
  echo "----------------------------------------------------------------------------"
done < "$1" 