<script setup>
// 1. IMPORTACIONES
// 'ref' es el equivalente al 'useState' de React. Nos permite crear variables reactivas.
import { ref } from 'vue'

// 2. VARIABLES DE ESTADO (REACTIVIDAD)
const prompt = ref('')          // Guarda el texto que escribe el usuario en la caja de texto
const responseText = ref('')    // Guarda la respuesta que nos devuelva FastAPI
const isLoading = ref(false)    // Controla si se muestra el indicador de "Cargando..." en pantalla

// 3. FUNCIONES DE CONECTIVIDAD (PETICIÓN AL BACKEND)
const generateContent = async () => {
  // Si el usuario no escribió nada, salimos de la función
  if (!prompt.value.trim()) return

  isLoading.value = true
  responseText.value = ''

  try {
    // Hacemos la llamada HTTP POST directa a nuestro servidor de FastAPI
    const response = await fetch('http://127.0.0.1:8000/api/v1/content/generate-text', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      // Convertimos los datos de JavaScript a un string JSON idéntico al que espera FastAPI
      body: JSON.stringify({
        prompt: prompt.value
      })
    })

    const data = await response.json()
    
    if (data.status === 'success') {
      // Guardamos el resultado de la IA en nuestra variable reactiva
      responseText.value = data.result
    } else {
      responseText.value = 'Hubo un problema al generar el contenido.'
    }
  } catch (error) {
    console.error('Error al conectar con el backend:', error)
    responseText.value = 'No se pudo conectar con el servidor de FastAPI.'
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div class="app-container">
    <h1>Agente de Contenido Bryan (Vue.js + FastAPI)</h1>
    <hr />

    <div class="input-section">
      <textarea 
        v-model="prompt" 
        placeholder="Escribe una instrucción para el agente de contenido..."
        rows="4"
      ></textarea>
      
      <button :disabled="isLoading" @click="generateContent">
        {{ isLoading ? 'Generando...' : 'Enviar a Bedrock' }}
      </button>
    </div>

    <div v-if="responseText" class="result-section">
      <h3>Respuesta del Agente:</h3>
      <p>{{ responseText }}</p>
    </div>
  </div>
</template>

<style scoped>
/* CSS scoped: Este diseño solo aplica a este componente */
.app-container {
  max-width: 600px;
  margin: 40px auto;
  font-family: Arial, sans-serif;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  resize: vertical;
}

button {
  margin-top: 10px;
  padding: 10px 20px;
  background-color: #41B883; /* El clásico verde de Vue.js */
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
}

button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.result-section {
  margin-top: 20px;
  padding: 15px;
  background-color: #f9f9f9;
  border-left: 4px solid #41B883;
}
</style>
