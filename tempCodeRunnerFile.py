   # --- Step 3: Roads (Challenge 2) -----------------------------------------
    try:
        from algorithms.road_network import build_roads
        cost = build_roads(graph)
        print(f"[C2]    Road network built — total cost: {cost:.1f}")
    except NotImplementedError as e:
        print(f"[C2]    {e}")