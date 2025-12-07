package com.pattern.patternApi.Controller;

import com.pattern.patternApi.Service.PatternService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/pattern")
public class PatternController {

    private final PatternService patternService;

    public PatternController(PatternService patternService) {
        this.patternService = patternService;
    }

    @PostMapping("/predict")
    public ResponseEntity<?> predictPattern(@RequestParam("file") MultipartFile file) {
        return ResponseEntity.ok(patternService.predict(file));
    }
}