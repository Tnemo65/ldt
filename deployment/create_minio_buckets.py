import boto3

s3 = boto3.client(
    's3',
    endpoint_url='http://ldt-minio:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin123',
    region_name='us-east-1'
)

buckets = [
    'cadqstream-raw',
    'cadqstream-violations',
    'cadqstream-anomalies',
    'cadqstream-metrics',
    'cadqstream-drift',
    'cadqstream-dlq'
]

for b in buckets:
    try:
        s3.head_bucket(Bucket=b)
        print(f'Exists: {b}')
    except Exception as e:
        try:
            s3.create_bucket(Bucket=b)
            print(f'Created: {b}')
        except Exception as e2:
            print(f'Error {b}: {e2}')

print()
print('All buckets:')
for b in s3.list_buckets()['Buckets']:
    name = b['Name']
    print(f'  {name}')
