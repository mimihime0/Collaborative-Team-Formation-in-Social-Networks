## this is with the auto run 
# I will tune the demon with flour values 
# 5,15,30,50
import networkx as nx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
import copy
class TeamFormation_Hybrid:
    def __init__(self,file_path,req_skills):
        self.file_path = file_path
        self.G=None #The social network graph
        self.teams_counter=0 # counter to help assign an id for any valid generated team 
        # its final value is the total number of teams in the whole run 
        self.req_skills=req_skills
        self.p = len(req_skills)
        self.estimated_longest_path=0


    def initial_setup(self):
        print(f"-----initial set up:-----\n")
        # load the graph
        self.G = nx.read_gml(self.file_path)
        print(f"Number of nodes: {self.G.number_of_nodes()}")
        # self.visualize_graph()
        self.print_data()
        experts_list = list(self.G.nodes)
        # Highest edge weight
        highest_weight = max(data['weight'] for u, v, data in self.G.edges(data=True))
        # Estimated longest path = infinity
        self.estimated_longest_path = highest_weight * self.G.number_of_nodes()
        print("Highest edge weight:", highest_weight)
        print("Estimated longest path:", self.estimated_longest_path)

        return experts_list    
    def print_data(self):
        print("\n=== Graph Information ===")
        
        # Print Nodes
        print("\n--- Nodes ---")
        for node_id, attributes in self.G.nodes(data=True):
            print(f"Node ID: {node_id}")
            print(f"  Label: {attributes.get('label', 'N/A')}")
            print(f"  Personnel Cost: {attributes.get('personnel_cost', 'N/A')}")
            skills = attributes.get("skills", [])
            print(f"  Skills: {', '.join(skills)}")
        
        # Print Edges
        print("\n--- Edges ---")
        for u, v, edge_attrs in self.G.edges(data=True):
            weight = edge_attrs.get("weight", "N/A")  # Default to 'N/A' if no weight exists
            print(f"Edge: {u} -> {v}, Weight: {weight}")
    # visualize the graph 
    def visualize_graph(self):
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(self.G, k=25.0)
        nx.draw(self.G, pos, with_labels=True, node_size=3000, node_color='lightblue', edge_color='gray', font_size=10)
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.title("Expert Graph Visualization")
        plt.show()
    
    # the Hybrid algorithm
    def hybrid_algorithm(self,experts_list, population_size=60, elite_sites=4, best_sites=6, best_moves=5):
        print(f" -----The Hybrid algorithm-----\n")
        iteration=0
        max_iterations=30
        # create the inital population
        population = self.generate_teams_randomly(population_size,experts_list)
        while(iteration<max_iterations):
            print(f"------Population {iteration+1}------\n")
            # Sort teams based on cost
            sorted_teams = sorted(population, key=lambda team: team["cost"])  # Ascending
            self.print_teams(sorted_teams)
        
            elite_teams = sorted_teams[:elite_sites]  # First elite_sites teams (lowest cost)
            best_teams = sorted_teams[elite_sites:elite_sites + best_sites]  # Next best_sites teams

            # Mutate elite teams with Demon Algorithm
            e=0
            jmax=5
            # demon_credit=20
            new_elite_teams = []
            for team in elite_teams:
                demon_credit=200
                j=1
                mutated_team=copy.deepcopy(team)
                e+= 1
                best_team=copy.deepcopy(team)
                print("\n----------------------\n")
                print(f"-{e}- original Elite Team ID: {team['tid']}, Cost: {team['cost']}, Skills: {team['experts']}\n")
                while (j<=jmax):
                    print(f"\nDemon Iteration {j} for Elite team -{e}-:\n")
                    self.mutate_team(mutated_team)
                    delta_E=mutated_team['cost']-team['cost']
                    if(delta_E<0):
                        j=1
                    else:
                        j+=1
                    print(f"delta_E={delta_E}")
                    if(delta_E<=demon_credit):
                        team=mutated_team
                        demon_credit=demon_credit-delta_E
                        print(f"demon_credit after accepting the new solution: {demon_credit}")
                        mutated_team=copy.deepcopy(team)
                        if team['cost'] < best_team['cost']:
                            best_team = copy.deepcopy(team)
                            print(f"The new best team ID: {best_team['tid']}, Cost: {best_team['cost']}, Skills: {best_team['experts']}\n")
                        print(f"\nOriginal Team -{e}- After The Demon Iteration {j}, Team ID: {team['tid']}, Cost: {team['cost']}, Skills: {team['experts']}\n")
                print(f"\nNumebr of iterations/trials without improvement for Elite Team -{e}-: {j-1}\n")
                print(f"\nThe improved original Elite Team -{e}- ID: {best_team['tid']}, Cost: {best_team['cost']}, Skills: {best_team['experts']}\n")
                new_elite_teams.append(best_team)

            b=0
            # Mutate best teams with fewer moves
            new_best_teams = []
            for team in best_teams:
                b+= 1
                original_team = copy.deepcopy(team)
                print("\n----------------------\n")
                print(f"-{b}- original Best Team ID: {original_team['tid']}, Cost: {original_team['cost']}, Skills: {original_team['experts']}\n")
                candidates = [original_team]     
                for i in range(best_moves):
                    print(f"Best Iteration {i+1}:")
                    mutated_team = self.mutate_team(copy.deepcopy(original_team))
                    candidates.append(mutated_team)
                    print(f"original Team -{b}- After The Best Iteration {i+1}, Team ID: {mutated_team['tid']}, Cost: {mutated_team['cost']}, Skills: {mutated_team['experts']}\n")
                best_candidate = min(candidates, key=lambda candidate: candidate["cost"])
                print(f"\nBest candidate of original Best Team -{b}- ID: {best_candidate['tid']}, Cost: {best_candidate['cost']}, Skills: {best_candidate['experts']}\n")
                new_best_teams.append(best_candidate)
            new_generated_teams = new_elite_teams + new_best_teams
            # Regenerate the rest of the population (replace the non-elite, non-best teams) and replace any duplications with random teams if any
            population = self.generate_teams_randomly(population_size,experts_list,new_generated_teams)  # Regenerate the remaining random teams
            iteration+=1
        print("-------FINAL POPULATION------")
        self.print_teams(population)
        suboptimal_team = min(population, key=lambda team: team["cost"])
        print(f"Suboptimal Team ID: {suboptimal_team['tid']}, Cost: {suboptimal_team['cost']}, Skills: {suboptimal_team['experts']}")
        print("----------------------\n")
        return suboptimal_team
    
    def print_teams(self,teams):
        if teams is not None:
            for team in teams:
                print(f"Team ID: {team['tid']}, Cost: {team['cost']}, Skills: {team['experts']}")
    
    def generate_teams_randomly(self,population_size,experts_list,new_generated_teams=None):
        # a list to store all teams for the current population generation 
        teams=[]
        
        max_attempts = population_size * 10  # to Prevent infinite loop, beacue the generated teams might be limted to a number
        # due (maybe) to the rareness of the skills
        attempts = 0
        # check the duplications between the elite and the best teams
        if(new_generated_teams != None):
            i = 0
            while i < len(new_generated_teams):
                
                j = i + 1
                while j < len(new_generated_teams):
                    if new_generated_teams[i]["experts"] == new_generated_teams[j]["experts"]:
                        new_generated_teams.pop(j)  # Remove the duplicate team
                    else:
                        j += 1  
                i+=1
            # inc the population_size to replace the duplicate teams with random teams later
            population_size=population_size-len(new_generated_teams)
        
        while len(teams) < population_size and attempts < max_attempts :
            attempts+=1
            team = self.generate_team_randomly(experts_list) 
            # Discard team if it does not cover all the skills
            if team is None:
                print("some skills have abs no expert to covered ")
                return None
            else:
                # Check if the new genrated team is unique within the teams list
                if not any(t["experts"] == team["experts"] for t in teams):
                    if (new_generated_teams==None) or not any(t["experts"] == team["experts"] for t in new_generated_teams) :
                        # calculate the obj func for the team
                        team["tid"] = self.teams_counter
                        team["cost"] = self.objective_function(team,lambda_val=0.2)
                        self.teams_counter+=1
                        # add the team to the teams list
                        teams.append(team)
        # if we are creating the initial population
        if (new_generated_teams== None): 
            return teams 
        else:
            new_generated_teams += teams
            return new_generated_teams
    # method generate only one valid team that satisfey the constrains 
    def generate_team_randomly(self,experts_list):
        # dict that represents a team  
        team = {
            "tid": None,
            "cost": 0,
            "experts": {}  # Placeholder for skills, key as the responsible expert id , the value is a set of skills
        }
        # this to introduce more randomness so that the experts adjacent to the current random_start_node_index
        # might not be for the next team generation
        random.shuffle(experts_list)

        #print("suffled :")
        #print("Shuffled nodes:", experts_list)
        #print("-------------------------------------------")


        for skill in self.req_skills:
            # Generate a random number between 0 and number of experts - 1 to pick a random expert
            random_start_node_index = random.randint(0, len(experts_list) - 1)
            for i in range(len(experts_list)):
                expert_id = experts_list[(random_start_node_index + i) % len(experts_list)]
                # Extract the skills of the picked expert
                expert_skills = self.G.nodes[str(expert_id)].get("skills", [])
                # check if the current skill is in the experts skills
                if skill in expert_skills:
                    # if the current skill is the first skill coverd by the current expert
                    # then initilize its set of skills with the current skill 
                    if expert_id not in team["experts"]:
                        team["experts"][expert_id] = {skill}
                        break
                    # if the current skill can be coverd by the current expert and he already cover some skills
                    # then we need to make sure he dosent cover all the req skills by ensuring 
                    # that the number of skills he covers is less than the number of the req skills-1 
                    elif len(team["experts"].get(expert_id))<self.p-1:
                        team["experts"][expert_id].add(skill)
                        break
            # the skill couldnt be covered by any expert 
            # i think any skill is at least covered by one expert based on paper 1, if im correct this line(and related lines) are not needed
                if i==len(experts_list)-1:
                    print(f"{skill} {i} has abs no expert to cover it from the graph")
                    return None
        return team    
    def objective_function(self,team,lambda_val=0.2):
        # personnel cost (pc)
        pc=0
        for expert_id in team["experts"]:
            pc += (self.G.nodes[str(expert_id)].get("personnel_cost", 0) * len(team["experts"].get(expert_id)))
        # communication cost (cc)
        expert_ids = list(team["experts"]) 
        cc = 0
        for i in range(len(expert_ids)):
            for j in range(i + 1, len(expert_ids)):
                cost = self.calculate_communication_cost( str(expert_ids[i]), str(expert_ids[j]))
                if cost == self.estimated_longest_path:
                    # debug: to check if truly the two experts have no path
                    print(f"infinte path in Team ID: {team['tid']}, expert1: {expert_ids[i]}, expert2: {expert_ids[j]}")
                
                cc += cost

        # Fitness calculation
        fitness = (self.p - 1) * (1 - lambda_val) * pc + 2 * lambda_val * cc

        return fitness
    def calculate_communication_cost(self,expert1, expert2):
        try:
            return nx.shortest_path_length(self.G, source= expert1, target=expert2, weight="weight")
        except nx.NetworkXNoPath:
            return self.estimated_longest_path  # no path exists
        
    def mutate_team(self,team):
        #why no loop ? 
        #if iteration_number >= 1:# ??
                experts = list(team["experts"].keys())
                print(f"orignal team (before mutaion): {experts}")
                # Randomly select an expert to replace
                expert_to_replace = random.choice(experts)
                expert_skills = team["experts"][expert_to_replace]

                print(f"Original Expert ID to replace: {expert_to_replace}, The skills covered by the expert to be replaced in the team: {expert_skills}")

                # Randomly choose one skill from the selected expert
                skill_to_replace = random.choice(list(expert_skills))
                print(f"Randomly selected skill to be replaced: {skill_to_replace}")
                # Find available experts with the same skill (exclude current team members)
                available_experts = [
                    expert_id for expert_id in self.G.nodes
                    if skill_to_replace in self.G.nodes[str(expert_id)].get("skills", [])
                    and expert_id != expert_to_replace 
                    and (expert_id not in team["experts"] or len(team["experts"][expert_id]) < self.p - 1)
                    # must do smth about this 
                    # if this line deleted then we must check that the number of experts coverd skills is less than p
                    #and expert_id not in team["experts"]

                ]
                print(f"Available experts with the same skill: {available_experts}")
                if available_experts:
                    new_expert = random.choice(available_experts)
                    if new_expert not in team["experts"]:
                    #this line was overwriting the exsited skills if the expert is already in the team dict 
                    # which means it will del the prev skills ( all of them ) and write the new skill in the set 
                    # so it will have only have one skill in it set of skills 
                        team["experts"][new_expert] = {skill_to_replace}
                    team["experts"][new_expert].add(skill_to_replace)
                    team["experts"][expert_to_replace].discard(skill_to_replace)
                    print(f"old expert (replaced expert) updated skills set: {team['experts'][expert_to_replace]}")
                    # If the expert has no skills left, we remove them from the team
                    if not team["experts"][expert_to_replace]:
                        del team["experts"][expert_to_replace]
                    team["cost"] = self.objective_function(team)
                else:
                    print(f"No available expert with matching skill for {skill_to_replace}.")
                return team
#-----------------Main-------------------------
def read_projects(file_path):
    """
    Reads a file containing projects and their required skills.

    Each line in the file should follow the format:
    Project X: Skill1, Skill2, Skill3, ...

    Returns:
        A list of projects, where each project is a list of skills.
    """
    projects = []

    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue

            # Split on colon to separate "Project X" from the skill list
            try:
                _, skills_str = line.strip().split(":", 1)
                # Split the skills string on commas and remove any extra whitespace
                skills = [skill.strip() for skill in skills_str.split(",")]
                projects.append(skills)
            except ValueError:
                print(f"Skipping malformed line: {line.strip()}")

    return projects

def run_multiple_bees_runs(file_path):
    all_suboptimal_teams = []
    projects = read_projects("projects_8_skills.txt")
    for i, project in enumerate(projects, start=1):
        print(f"Project {i}: {project}")
        print(f"\n========== Run/project No. {i} ==========\n")
        team_formation_hybrid = TeamFormation_Hybrid(file_path, project)
        experts_list = team_formation_hybrid.initial_setup()
        suboptimal_team = team_formation_hybrid.hybrid_algorithm(experts_list)
        all_suboptimal_teams.append(suboptimal_team)

    print("\n======= Summary of All Runs/projects (Suboptimal Teams) =======\n")
    total_cost = 0
    for i, team in enumerate(all_suboptimal_teams, 1):
        print(f"Run {i}: Team ID: {team['tid']}, Cost: {team['cost']}, Skills: {team['experts']}")
        total_cost += team['cost']

    average_cost = total_cost / 50
    print(f"\n⭐ Average Suboptimal Team Cost across {50} runs/projects: {average_cost:.2f}")

def main():
    file_path = 'experts250_graph.gml'
    # req_skills=["SQL","Java","Python","React"]
    # team_formation_hybrid = TeamFormation_Hybrid(file_path,req_skills)
    # experts_list=team_formation_hybrid.initial_setup()
    # team_formation_hybrid.hybrid_algorithm(experts_list)
    run_multiple_bees_runs(file_path) 

if __name__ == "__main__":
    main()
