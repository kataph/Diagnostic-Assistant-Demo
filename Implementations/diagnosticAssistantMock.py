from asyncio import sleep

from environment_classes import DiagnosticAction, DiagnosticAssistant, Observation


class DiagnosticAssistantMock(DiagnosticAssistant):
    
    async def setup(self, observations: list[Observation]) -> None:
        self.logger.info("The mock diagnostic assistant is waking up...")
        await sleep(1)
        
    async def suggest_action(self) -> DiagnosticAction:
        self.logger.info("The mock diagnostic assistant is thinking...")
        await sleep(1)
        suggested_action = DiagnosticAction(type = "Adjust", target = str(self.configuration.SYSTEM_URL), description = "Have you tried turning off and on?")
        self.logger.info(f"It suggested {suggested_action.get_name()}...")
        return suggested_action