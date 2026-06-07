from pygrabber.dshow_graph import FilterGraph

def list_cameras_with_names():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    
    print("--- Available Cameras ---")
    if not devices:
        print("No cameras found!")
    else:
        for index, name in enumerate(devices):
            print(f"ID: {index} | Name: {name}")

if __name__ == "__main__":
    list_cameras_with_names()