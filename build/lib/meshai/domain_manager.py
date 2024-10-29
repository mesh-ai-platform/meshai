# meshai/domain_manager.py

class DomainManager:
    """
    Manages domains and associates models with domains.
    """
    def __init__(self):
        self.domains = {}

    def create_domain(self, domain_name):
        """
        Creates a new domain.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = {'models': []}
            print(f"Domain '{domain_name}' created.")
        else:
            print(f"Domain '{domain_name}' already exists.")

    def assign_model_to_domain(self, domain_name, model_handler):
        """
        Assigns a model handler to a domain.
        """
        if domain_name in self.domains:
            self.domains[domain_name]['models'].append(model_handler)
            print(f"Model assigned to domain '{domain_name}'.")
        else:
            raise ValueError(f"Domain '{domain_name}' does not exist.")

    def list_domains(self):
        """
        Lists all domains.
        """
        return list(self.domains.keys())

    def list_models_in_domain(self, domain_name):
        """
        Lists all models in a domain.
        """
        if domain_name in self.domains:
            return self.domains[domain_name]['models']
        else:
            raise ValueError(f"Domain '{domain_name}' does not exist.")
