-- Создание всех таблиц проекта Smart Restock Advisor
-- Таблицы создаются в правильном порядке (сначала базовые, затем зависимые)

-- 1. Пользователи системы
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Поставщики
CREATE TABLE IF NOT EXISTS suppliers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    contact_person VARCHAR(100),
    phone VARCHAR(20),
    email VARCHAR(100),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Клиенты
CREATE TABLE IF NOT EXISTS clients (
    id SERIAL PRIMARY KEY,
    client_code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Товары/Продукты
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,
    descr VARCHAR(255) NOT NULL,
    article VARCHAR(100),
    measure VARCHAR(20),
    nds_rate DECIMAL(5,2) DEFAULT 20.00,
    is_folder BOOLEAN DEFAULT false,
    parent_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Заказы
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_number VARCHAR(50) UNIQUE NOT NULL,
    supplier_id INTEGER REFERENCES suppliers(id),
    product_code VARCHAR(50) REFERENCES products(code),
    quantity DECIMAL(10,2) NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_amount DECIMAL(10,2),
    order_date DATE DEFAULT CURRENT_DATE,
    expected_delivery DATE,
    status VARCHAR(20) DEFAULT 'pending',
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Продажи
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    product_code VARCHAR(50) REFERENCES products(code),
    client_code VARCHAR(50) REFERENCES clients(client_code),
    earned DECIMAL(10,2) NOT NULL,
    value INTEGER NOT NULL,
    date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Остатки на складе
CREATE TABLE IF NOT EXISTS stock_levels (
    id SERIAL PRIMARY KEY,
    product_code VARCHAR(50) REFERENCES products(code),
    value DECIMAL(10,2) NOT NULL,
    date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- ТЕСТОВЫЕ ДАННЫЕ
-- ============================================

-- Пользователи (пароль: admin123 / manager123)
INSERT INTO users (username, email, password_hash, full_name, role) VALUES
('admin', 'admin@warehouse.com', 'pbkdf2:sha256:260000$abc123$def456', 'Администратор', 'admin'),
('manager', 'manager@warehouse.com', 'pbkdf2:sha256:260000$xyz789$uvw012', 'Менеджер Склада', 'manager')
ON CONFLICT (username) DO NOTHING;

-- Поставщики
INSERT INTO suppliers (name, contact_person, phone, email, address) VALUES
('ООО "Поставщик 1"', 'Иванов Иван', '+79991234567', 'supplier1@mail.com', 'г. Москва, ул. Ленина, д. 1'),
('ООО "Поставщик 2"', 'Петров Петр', '+79997654321', 'supplier2@mail.com', 'г. Санкт-Петербург, ул. Пушкина, д. 10')
ON CONFLICT DO NOTHING;

-- Клиенты
INSERT INTO clients (client_code, name) VALUES
('CL001', 'ООО "Клиент 1"'),
('CL002', 'ИП Сидоров'),
('CL003', 'ООО "Розничная сеть"')
ON CONFLICT (client_code) DO NOTHING;

-- Товары
INSERT INTO products (code, descr, article, measure, nds_rate, is_folder) VALUES
('P001', 'Ноутбук Lenovo IdeaPad', 'NLP-001', 'шт.', 20.00, false),
('P002', 'Мышь беспроводная', 'MB-202', 'шт.', 20.00, false),
('P003', 'Клавиатура механическая', 'KM-305', 'шт.', 20.00, false),
('P004', 'Монитор 24 дюйма', 'MON-24', 'шт.', 20.00, false),
('P005', 'Наушники Bluetooth', 'NB-501', 'шт.', 20.00, false),
('P006', 'Веб-камера Full HD', 'WC-1080', 'шт.', 20.00, false),
('P007', 'Внешний жесткий диск 1TB', 'HDD-1TB', 'шт.', 20.00, false),
('P008', 'Флеш-накопитель 128GB', 'USB-128', 'шт.', 20.00, false)
ON CONFLICT (code) DO NOTHING;

-- Остатки на складе (история за 3 периода)
INSERT INTO stock_levels (product_code, value, date) VALUES
-- Неделю назад
('P001', 15, CURRENT_DATE - INTERVAL '7 days'),
('P002', 45, CURRENT_DATE - INTERVAL '7 days'),
('P003', 22, CURRENT_DATE - INTERVAL '7 days'),
('P004', 8, CURRENT_DATE - INTERVAL '7 days'),
('P005', 30, CURRENT_DATE - INTERVAL '7 days'),
('P006', 18, CURRENT_DATE - INTERVAL '7 days'),
('P007', 12, CURRENT_DATE - INTERVAL '7 days'),
('P008', 25, CURRENT_DATE - INTERVAL '7 days'),
-- 3 дня назад
('P001', 12, CURRENT_DATE - INTERVAL '3 days'),
('P002', 38, CURRENT_DATE - INTERVAL '3 days'),
('P003', 18, CURRENT_DATE - INTERVAL '3 days'),
('P004', 5, CURRENT_DATE - INTERVAL '3 days'),
('P005', 25, CURRENT_DATE - INTERVAL '3 days'),
('P006', 15, CURRENT_DATE - INTERVAL '3 days'),
('P007', 9, CURRENT_DATE - INTERVAL '3 days'),
('P008', 20, CURRENT_DATE - INTERVAL '3 days'),
-- Сегодня
('P001', 10, CURRENT_DATE),
('P002', 32, CURRENT_DATE),
('P003', 15, CURRENT_DATE),
('P004', 3, CURRENT_DATE),
('P005', 22, CURRENT_DATE),
('P006', 12, CURRENT_DATE),
('P007', 7, CURRENT_DATE),
('P008', 18, CURRENT_DATE)
ON CONFLICT DO NOTHING;

-- Продажи (история за 30 дней)
INSERT INTO sales (product_code, client_code, earned, value, date) VALUES
-- 30 дней назад
('P001', 'CL001', 50000.00, 1, CURRENT_DATE - INTERVAL '30 days'),
('P002', 'CL002', 1500.00, 3, CURRENT_DATE - INTERVAL '28 days'),
-- 20 дней назад
('P003', 'CL003', 4500.00, 3, CURRENT_DATE - INTERVAL '20 days'),
('P004', 'CL001', 25000.00, 1, CURRENT_DATE - INTERVAL '18 days'),
-- 10 дней назад
('P005', 'CL002', 8000.00, 4, CURRENT_DATE - INTERVAL '10 days'),
('P006', 'CL003', 4500.00, 3, CURRENT_DATE - INTERVAL '8 days'),
-- 5 дней назад
('P001', 'CL002', 52000.00, 1, CURRENT_DATE - INTERVAL '5 days'),
('P007', 'CL001', 12000.00, 2, CURRENT_DATE - INTERVAL '4 days'),
-- 2 дня назад
('P008', 'CL002', 3000.00, 5, CURRENT_DATE - INTERVAL '2 days'),
('P002', 'CL003', 1200.00, 2, CURRENT_DATE - INTERVAL '1 day'),
-- Сегодня
('P003', 'CL001', 4800.00, 3, CURRENT_DATE),
('P005', 'CL003', 6000.00, 3, CURRENT_DATE)
ON CONFLICT DO NOTHING;

-- Заказы (активные и выполненные)
INSERT INTO orders (order_number, supplier_id, product_code, quantity, unit_price, total_amount, expected_delivery, status) VALUES
-- Активные заказы
('ORD-2024-1001', 1, 'P001', 5.00, 45000.00, 225000.00, CURRENT_DATE + INTERVAL '7 days', 'confirmed'),
('ORD-2024-1002', 2, 'P002', 20.00, 1500.00, 30000.00, CURRENT_DATE + INTERVAL '5 days', 'pending'),
('ORD-2024-1003', 1, 'P004', 3.00, 25000.00, 75000.00, CURRENT_DATE + INTERVAL '10 days', 'processing'),
-- Выполненные заказы
('ORD-2024-0999', 2, 'P003', 15.00, 1800.00, 27000.00, CURRENT_DATE - INTERVAL '2 days', 'delivered'),
('ORD-2024-0998', 1, 'P005', 10.00, 2500.00, 25000.00, CURRENT_DATE - INTERVAL '5 days', 'delivered')
ON CONFLICT (order_number) DO NOTHING;

-- Индексы для ускорения запросов
CREATE INDEX IF NOT EXISTS idx_products_code ON products(code);
CREATE INDEX IF NOT EXISTS idx_sales_product_code ON sales(product_code);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);
CREATE INDEX IF NOT EXISTS idx_stock_levels_product_code ON stock_levels(product_code);
CREATE INDEX IF NOT EXISTS idx_stock_levels_date ON stock_levels(date);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_expected_delivery ON orders(expected_delivery);

-- Сообщение об успешной инициализации
DO $$
BEGIN
    RAISE NOTICE 'База данных успешно инициализирована с тестовыми данными';
    RAISE NOTICE 'Товаров: %', (SELECT COUNT(*) FROM products);
    RAISE NOTICE 'Продаж: %', (SELECT COUNT(*) FROM sales);
    RAISE NOTICE 'Остатков: %', (SELECT COUNT(*) FROM stock_levels);
    RAISE NOTICE 'Заказов: %', (SELECT COUNT(*) FROM orders);
END $$;