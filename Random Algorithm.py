import networkx as nx
import matplotlib.pyplot as plt
import random
import copy
import pandas as pd
import os

class TeamFormation_BA:
    def __init__(self,file_path,req_skills):
        self.file_path = file_path
        self.G=None
        self.teams_counter=0 
        self.req_skills=req_skills
        self.p = len(req_skills)
        self.estimated_longest_path=0

    def initial_setup(self):
        print(f"-----initial set up:-----\n")
        self.G = nx.read_gml(self.file_path)
        print(f"Number of nodes: {self.G.number_of_nodes()}")
        experts_list = list(self.G.nodes)
        
        highest_weight = 0
        if self.G.number_of_nodes() > 0 and self.G.edges(data=True):
            edge_weights = [data.get('weight', 0) for u, v, data in self.G.edges(data=True) if data.get('weight') is not None]
            if edge_weights:
                highest_weight = max(edge_weights)
        
        self.estimated_longest_path = highest_weight * self.G.number_of_nodes() if self.G.number_of_nodes() > 0 else 0
        print("Highest edge weight:", highest_weight)
        print("Estimated longest path:", self.estimated_longest_path)
        return experts_list

    def print_data(self):
        print("\n=== Graph Information ===")
        print("\n--- Nodes ---")
        for node_id, attributes in self.G.nodes(data=True):
            print(f"Node ID: {node_id}")
            print(f"  Label: {attributes.get('label', 'N/A')}")
            print(f"  Personnel Cost: {attributes.get('personnel_cost', 'N/A')}")
            skills = attributes.get("skills", [])
            print(f"  Skills: {', '.join(skills)}")
        print("\n--- Edges ---")
        for u, v, edge_attrs in self.G.edges(data=True):
            weight = edge_attrs.get("weight", "N/A")
            print(f"Edge: {u} -> {v}, Weight: {weight}")

    def visualize_graph(self):
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(self.G, k=25.0)
        nx.draw(self.G, pos, with_labels=True, node_size=3000, node_color='lightblue', edge_color='gray', font_size=10)
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.title("Expert Graph Visualization")

    def find_best_team_iteratively(self, experts_list, num_unique_teams_to_evaluate=300):
        print(f"\n----- Iteratively evaluating team generations to find the best one (aiming for {num_unique_teams_to_evaluate} unique teams) -----")
        
        best_team_overall = None
        min_cost_overall = float('inf')
        
        unique_teams_considered = 0
        generation_attempts = 0

        max_generation_attempts = num_unique_teams_to_evaluate * 10 

        seen_team_compositions = set() 

        while unique_teams_considered < num_unique_teams_to_evaluate and generation_attempts < max_generation_attempts:
            generation_attempts += 1
            
            if generation_attempts % (max_generation_attempts // 10) == 0 and generation_attempts > 0 : 
                 print(f"  Attempt {generation_attempts}/{max_generation_attempts}, Unique teams found so far: {unique_teams_considered}")

            current_team_structure = self.generate_team_randomly(experts_list)

            if current_team_structure is None:
                continue

            expert_composition_tuple = frozenset(
                (expert_id, frozenset(skills)) for expert_id, skills in current_team_structure["experts"].items()
            )


            if expert_composition_tuple in seen_team_compositions:
                continue 
            
            seen_team_compositions.add(expert_composition_tuple)

            current_team_object = {
                "tid": f"T{self.teams_counter}",
                "cost": 0,
                "experts": current_team_structure["experts"]
            }
            current_team_object["cost"] = self.objective_function(current_team_object, lambda_val=0.2)
            self.teams_counter += 1
            
            unique_teams_considered += 1

            if current_team_object["cost"] < min_cost_overall:
                min_cost_overall = current_team_object["cost"]
                best_team_overall = copy.deepcopy(current_team_object) 


        print(f"\nTotal generation attempts: {generation_attempts}")
        print(f"Total unique valid teams considered: {unique_teams_considered}")

        if unique_teams_considered < num_unique_teams_to_evaluate and unique_teams_considered > 0:
            print(f"Warning: Could only find and evaluate {unique_teams_considered} unique valid teams (requested {num_unique_teams_to_evaluate}).")
        elif unique_teams_considered == 0:
             print("Critical: No unique valid teams could be generated.")


        if best_team_overall:
            print("\n------ Best Team Found Iteratively ------")
            print(f"Team ID: {best_team_overall['tid']}")
            print(f"Cost: {best_team_overall['cost']:.2f}")
            print(f"Experts and their covered skills:")
            for expert_id, skills_covered in best_team_overall['experts'].items():
                print(f"  Expert {expert_id}: {', '.join(sorted(list(skills_covered)))}")
            print("---------------------------------------\n")
        else:
            print("No valid teams could be generated that meet the criteria.")
            
        return best_team_overall

    def generate_team_randomly(self, experts_list):
        team_experts_structure = {} 
        
        covered_skills_overall = set()
        shuffled_req_skills = random.sample(self.req_skills, len(self.req_skills))

        for skill_to_cover in shuffled_req_skills:
            if skill_to_cover in covered_skills_overall:
                continue

            potential_experts_for_skill = [
                expert_id for expert_id in experts_list
                if skill_to_cover in self.G.nodes[str(expert_id)].get("skills", [])
            ]
            random.shuffle(potential_experts_for_skill)

            skill_assigned_this_iteration = False
            for expert_id in potential_experts_for_skill:
                num_skills_already_by_this_expert = len(team_experts_structure.get(expert_id, set()))

                if self.p > 1 and num_skills_already_by_this_expert >= (self.p - 1):
                    continue 

                if expert_id not in team_experts_structure:
                    team_experts_structure[expert_id] = set()
                
                team_experts_structure[expert_id].add(skill_to_cover)
                covered_skills_overall.add(skill_to_cover)
                skill_assigned_this_iteration = True
                break 
            
            if not skill_assigned_this_iteration:
                return None 

        if len(covered_skills_overall) != len(self.req_skills):
            return None
            
        return {"experts": team_experts_structure}


    def objective_function(self,team,lambda_val=0.2):
        pc=0
        for expert_id in team["experts"]:
            pc += (self.G.nodes[str(expert_id)].get("personnel_cost", 0) * len(team["experts"].get(expert_id)))

        expert_ids_in_team = list(team["experts"])
        cc = 0
        if len(expert_ids_in_team) > 1:
            for i in range(len(expert_ids_in_team)):
                for j in range(i + 1, len(expert_ids_in_team)):
                    cost = self.calculate_communication_cost( str(expert_ids_in_team[i]), str(expert_ids_in_team[j]))
                    cc += cost
        
        if self.p == 1:
            return pc 
        else:
            fitness = (self.p - 1) * (1 - lambda_val) * pc + 2 * lambda_val * cc
            return fitness


    def calculate_communication_cost(self,expert1, expert2):

        try:
            if self.G is None: raise ValueError("Graph G not initialized.")
            return nx.shortest_path_length(self.G, source= expert1, target=expert2, weight="weight")
        except nx.NetworkXNoPath:
            return self.estimated_longest_path
        except nx.NodeNotFound:
            print(f"Error: Node not found in graph. Expert1: {expert1}, Expert2: {expert2}")
            return self.estimated_longest_path


def run_skillset_file_once_per_project(skill_file_path, graph_file_path):
    results = []
    with open(skill_file_path, 'r') as file: lines = file.readlines()

    for i, line in enumerate(lines):
        if ':' not in line: continue
        skill_list_str = line.split(':', 1)[1].strip()
        req_skills = [skill.strip() for skill in skill_list_str.split(',')]
        print(f"\n🔁 Running Project {i + 1} with skills: {req_skills}")

        tf = TeamFormation_BA(graph_file_path, req_skills)
        experts = tf.initial_setup()
        
        best_team_found = None
        try:
            best_team_found = tf.find_best_team_iteratively(experts, num_unique_teams_to_evaluate=300)
        except Exception as e:
            print(f"❌ Skipping Project {i + 1}: An error occurred - {e}")
            results.append({
                'Project': i + 1, 'Skills': ', '.join(req_skills),
                'Cost': 'N/A', 'Team': f'Error: {e}'})
            continue

        if best_team_found:
            team_str = ', '.join([f"{eid}: {{{', '.join(sorted(list(skills)))}}}" for eid, skills in best_team_found['experts'].items()])
            results.append({
                'Project': i + 1, 'Skills': ', '.join(req_skills),
                'Cost': f"{best_team_found['cost']:.2f}", 'Team': team_str})
        else:
            print(f"❌ Project {i + 1}: No valid team formed after all attempts.")
            results.append({
                'Project': i + 1, 'Skills': ', '.join(req_skills),
                'Cost': 'N/A', 'Team': 'No valid team formed'})
    
    df = pd.DataFrame(results)
    df['Cost_numeric'] = pd.to_numeric(df['Cost'], errors='coerce')
    valid_costs = df['Cost_numeric'].dropna()
    avg_cost = valid_costs.mean() if not valid_costs.empty else 0

    print("\n📊 ====== Final Results Table ======")
    print(df[['Project', 'Skills', 'Cost', 'Team']].to_string(index=False))
    print(f"\n✅ Average Cost (only valid projects): {avg_cost:.2f}")

    try: 
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.exists(desktop_path): os.makedirs(desktop_path)
        save_path = os.path.join(desktop_path, "project_results.csv")
        df[['Project', 'Skills', 'Cost', 'Team']].to_csv(save_path, index=False)
        print(f"\n📁 Table saved to: {save_path}")
    except Exception as e:
        print(f"\n❌ Error saving file to Desktop: {e}")
        fallback_path = "project_results.csv"
        try:
            df[['Project', 'Skills', 'Cost', 'Team']].to_csv(fallback_path, index=False)
            print(f"⚠️ File saved to current directory instead: {os.path.abspath(fallback_path)}")
        except Exception as e_fallback:
            print(f"❌❌ Critical Error saving file: {e_fallback}")
    return df, avg_cost

#-----------------Main-------------------------
def main():
    script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    project_file_name = "projects_10_skills.txt"
    graph_file_name = 'experts60_graph.gml'
    project_file_path = os.path.join(script_dir, project_file_name)
    graph_file_path = os.path.join(script_dir, graph_file_name)

    if not os.path.exists(project_file_path):
        print(f"Error: Project file not found at {project_file_path}")
        return
    if not os.path.exists(graph_file_path):
        print(f"Error: Graph file not found at {graph_file_path}")
        return
        
    print(f"Using project file: {project_file_path}")
    print(f"Using graph file: {graph_file_path}")
    run_skillset_file_once_per_project(project_file_path, graph_file_path)

if __name__ == "__main__":
    main()