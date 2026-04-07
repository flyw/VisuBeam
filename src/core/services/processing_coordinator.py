from src.core.audio.buffer import AudioChunk
from src.doa.services.doa_service import DOAService
from src.enhancement.services.enhancement_service import EnhancementService

class ProcessingCoordinator:
    def __init__(self, doa_service: DOAService, enhancement_service: EnhancementService):
        """
        Coordinates the processing flow between DOAService and EnhancementService.
        """
        self.doa_service = doa_service
        self.enhancement_service = enhancement_service

    def process_audio(self, audio_chunk: AudioChunk):
        """
        Processes an audio chunk:
        1. DOAService calculates DOA angles.
        2. EnhancementService uses these angles for MVDR/DTLN.
        """
        # 1. Get DOA results and processed audio (WPE/MCRA)
        doa_results, processed_data = self.doa_service.process_audio(audio_chunk)
        
        # Create a new chunk with processed data for enhancement service
        processed_chunk = audio_chunk.copy()
        processed_chunk.data = processed_data
        
        # 2. Pass processed audio and DOA results to EnhancementService
        self.enhancement_service.process_audio(processed_chunk, doa_results)

    def close(self):
        """Closes both services."""
        if self.doa_service:
            self.doa_service.close()
        if self.enhancement_service:
            self.enhancement_service.close()
