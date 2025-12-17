//Графики (болванки)
class ChartManager {
    constructor() {
        this.charts = new Map();
    }

    initForecastChart() {
        const canvas = document.getElementById('forecastChart');
        if (!canvas) {
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Данные для графика
        const dates = ['01.12', '02.12', '03.12', '04.12', '05.12', '06.12', '07.12', '08.12', '09.12', '10.12', '11.12', '12.12', '13.12', '14.12'];
        const salesData = [45, 52, 49, 60, 55, 58, 62, null, null, null, null, null, null, null];
        const forecastData = [null, null, null, null, null, null, 62, 65, 68, 72, 75, 78, 70, 65];
        const stockData = [180, 160, 140, 110, 85, 60, 35, null, null, null, null, null, null, null];
        const stockForecast = [null, null, null, null, null, null, 35, 20, 0, 0, 0, 0, 0, 0];
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Исторические продажи',
                        data: salesData,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Прогноз спроса',
                        data: forecastData,
                        borderColor: '#2980b9',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointStyle: 'circle',
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Исторические остатки',
                        data: stockData,
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Прогноз остатков',
                        data: stockForecast,
                        borderColor: '#e74c3c',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointStyle: 'circle',
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Минимальный запас',
                        data: Array(14).fill(20),
                        borderColor: '#e74c3c',
                        borderWidth: 2,
                        borderDash: [3, 3],
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Количество единиц'
                        }
                    }
                }
            }
        });
        
        this.charts.set('forecast', chart);
    }

    initAnalyticsCharts() {
        
        // Продажи по категориям
        const salesCtx = document.getElementById('salesByCategoryChart');
        if (salesCtx) {
            try {
                const chart = new Chart(salesCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Энергетики', 'Мороженное', 'Напитки', 'Чипсы', 'Аксессуары'],
                        datasets: [{
                            label: 'Продажи (шт.)',
                            data: [1250, 890, 1560, 540, 1320], 
                            backgroundColor: [
                                'rgba(52, 152, 219, 0.8)',
                                'rgba(46, 204, 113, 0.8)',
                                'rgba(155, 89, 182, 0.8)',
                                'rgba(241, 196, 15, 0.8)',
                                'rgba(230, 126, 34, 0.8)'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
                this.charts.set('salesByCategory', chart);
            } catch (error) {
                console.error('Error initializing sales chart:', error);
            }
        }

        // ABC-анализ
        const abcCtx = document.getElementById('abcAnalysisChart');
        if (abcCtx) {
            try {
                const chart = new Chart(abcCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Категория A (70%)', 'Категория B (20%)', 'Категория C (10%)'],
                        datasets: [{
                            data: [70, 20, 10], 
                            backgroundColor: [
                                'rgba(46, 204, 113, 0.8)',
                                'rgba(52, 152, 219, 0.8)',
                                'rgba(241, 196, 15, 0.8)'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
                this.charts.set('abcAnalysis', chart);
            } catch (error) {
                console.error('Error initializing ABC chart:', error);
            }
        }

        // Сезонность 
        const seasonCtx = document.getElementById('seasonalityChart');
        if (seasonCtx) {
            try {
                const chart = new Chart(seasonCtx, {
                    type: 'line',
                    data: {
                        labels: ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'],
                        datasets: [{
                            label: 'Продажи (шт.)',
                            data: [320, 350, 380, 420, 480, 520, 580, 560, 500, 450, 380, 350], 
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            borderWidth: 3,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
                this.charts.set('seasonality', chart);
            } catch (error) {
                console.error('Error initializing seasonality chart:', error);
            }
        }
    }

    initAllCharts() {
        this.initForecastChart();
        this.initAnalyticsCharts();
    }
}

// Обработчики событий
function initEventHandlers() {
    // Кнопки "Заказать"
    document.querySelectorAll('.btn-order').forEach(button => {
        button.addEventListener('click', function() {
            const product = this.closest('tr').querySelector('td:first-child').textContent;
            alert(`Заказ на ${product} оформлен!`);
            this.textContent = 'Заказано';
            this.disabled = true;
            this.classList.remove('btn-order');
            this.classList.add('btn-skip');
        });
    });

    // Кнопки "Пропустить"
    document.querySelectorAll('.btn-skip').forEach(button => {
        button.addEventListener('click', function() {
            const product = this.closest('tr').querySelector('td:first-child').textContent;
            alert(`Рекомендация для ${product} пропущена!`);
            this.textContent = 'Пропущено';
            this.disabled = true;
        });
    });

    // Обновление данных
    const updateBtn = document.querySelector('.btn-primary');
    if (updateBtn && updateBtn.textContent.includes('Обновить')) {
        updateBtn.addEventListener('click', function() {
            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Обновление...';
            this.disabled = true;
            
            setTimeout(() => {
                this.innerHTML = originalText;
                this.disabled = false;
                alert('Данные успешно обновлены!');
            }, 1500);
        });
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const chartManager = new ChartManager();
    chartManager.initAllCharts();
    initEventHandlers();
});

// Функции для работы с поставщиками
function viewSupplier(id) {
    alert('Просмотр поставщика ID: ' + id);
    // window.location.href = `/suppliers/${id}`;
}

function editSupplier(id) {
    alert('Редактирование поставщика ID: ' + id);
    // window.location.href = `/suppliers/${id}/edit`;
}

function deleteSupplier(id) {
    if (confirm('Удалить поставщика?')) {
        // Отправка DELETE запроса
        fetch(`/suppliers/${id}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => {
            if (response.ok) {
                alert('Поставщик удален');
                location.reload();
            } else {
                alert('Ошибка при удалении');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Ошибка сети');
        });
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Поиск в таблице
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('.suppliers-table tbody tr');

            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchTerm) ? '' : 'none';
            });
        });
    }

    // Добавление нового поставщика
    const addButton = document.querySelector('.btn-add');
    if (addButton) {
        addButton.addEventListener('click', function() {
            alert('Добавление нового поставщика');
            // window.location.href = '/suppliers/new';
        });
    }

    // Фильтрация по статусу
    const filterSelects = document.querySelectorAll('.filter-select');
    filterSelects.forEach(select => {
        select.addEventListener('change', function() {
            applyFilters();
        });
    });
});

// Применение фильтров
function applyFilters() {
    // Здесь можно реализовать фильтрацию через AJAX запрос к серверу
    console.log('Фильтры применены');
}

// Экспорт таблицы в Excel
function exportToExcel() {
    alert('Экспорт в Excel');
    // Реализация экспорта
}

// Обновление статуса поставщика
function updateSupplierStatus(supplierId, status) {
    fetch(`/suppliers/${supplierId}/status`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: status })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Статус обновлен');
            location.reload();
        }
    });
}

function refreshOrders() {
    location.reload();
}

function applyFilters() {
    const statusFilter = document.getElementById('statusFilter').value;
    const supplierFilter = document.getElementById('supplierFilter').value;
    const dateFrom = document.getElementById('dateFrom').value;
    const dateTo = document.getElementById('dateTo').value;

    const rows = document.querySelectorAll('.orders-table tbody tr');

    rows.forEach(row => {
        let show = true;

        // Фильтр по статусу
        if (statusFilter && row.getAttribute('data-status') !== statusFilter) {
            show = false;
        }

        // Фильтр по поставщику
        if (supplierFilter && row.getAttribute('data-supplier') !== supplierFilter) {
            show = false;
        }

        row.style.display = show ? '' : 'none';
    });
}

function clearFilters() {
    document.getElementById('statusFilter').value = '';
    document.getElementById('supplierFilter').value = '';
    document.getElementById('dateFrom').value = '';
    document.getElementById('dateTo').value = '';

    const rows = document.querySelectorAll('.orders-table tbody tr');
    rows.forEach(row => {
        row.style.display = '';
    });
}

// Инициализация при загрузке
document.addEventListener('DOMContentLoaded', function() {
    // Автоматически устанавливаем даты
    const today = new Date().toISOString().split('T')[0];
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    const weekAgoStr = weekAgo.toISOString().split('T')[0];

    document.getElementById('dateFrom').value = weekAgoStr;
    document.getElementById('dateTo').value = today;
});