"""
 Script to test postgres database read write.
 
 Steps to setup postgres and pgvector using docker.
 -------------------------------------------------
 ## Download image
 - docker pull pgvector/pgvector:pg17
 - docker images
 
 ## Create Volume
 - docker volume ls
 - docker volume create pgvector-data

 ## Create container
 - docker run --name pgvector-container -e POSTGRES_PASSWORD=postgres -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data -d pgvector/pgvector:pg17
 - docker ps
 
 ## pgadmin4
 - docker pull dpage/pgadmin4
 - docker run --name pgadmin-container -p 5050:80 -e PGADMIN_DEFAULT_EMAIL=user@domain.com -e PGADMIN_DEFAULT_PASSWORD=password -d dpage/pgadmin4
"""

import psycopg2

conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="postgres", port=5432)

cur = conn.cursor()

#Create DB table 
cur.execute(""" CREATE TABLE IF NOT EXISTS person (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    age INT,
    gender CHAR
);
""")

cur.execute("DROP TABLE IF EXISTS ITEMS")
# Insert records.
cur.execute("""
           INSERT INTO person(id, name, age, gender) VALUES
           (10, 'John', 30, 'm'), 
           (2, 'Amar', 40, 'm'), 
           (3, 'Dharmendar', 35, 'm'), 
           (4, 'Kromer', 56, 'm'), 
           (5, 'Jenny', 20, 'f'), 
           (6, 'Godavari', 43, 'f') ;
            """)



res = cur.execute("""SELECT * from person WHERE age >=31; """)

#for row in cur.fetchall():
#    print(row)
#

sql = cur.mogrify(""" 
                  SELECT * FROM person where starts_with(name, %s) and age > %s;
                  """, ("J",3))

print(sql)


conn.commit()
cur.close()



conn.close()