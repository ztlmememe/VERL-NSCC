import sys, ray

try:
    ray.init(logging_level="debug", address=sys.argv[1],  _node_ip_address=sys.argv[1].split(':')[0])

    print("\n=== Ray Cluster Status ===")
    print(f"Number of nodes: {len(ray.nodes())}")
    for node in ray.nodes():
        print("Node: {}, Status: {}".format(node["NodeManagerHostname"], node["Alive"]))

    ray.shutdown()

    print("Ray initialization successful!")
except Exception as e:
    print(f"Ray initialization failed: {str(e)}")
